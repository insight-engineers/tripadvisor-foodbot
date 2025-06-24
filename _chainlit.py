import asyncio

import chainlit as cl
import chainlit.data as cl_data
import yaml

from src.chat.chainlit import AskActionMessage
from src.chat.utils import generate_conv_summary, generate_next_response, generate_streaming_response
from src.helper.utils import encode_b64_string, get_admin_account, get_config_file, get_display_name, normalize_weights
from src.main import get_chat_settings, init_user_session, remove_next_response_actions


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    session_username, session_password = encode_b64_string(username), encode_b64_string(password)
    admin_username, admin_password = get_admin_account()

    if (session_username, session_password) == (admin_username, admin_password):
        return cl.User(
            identifier=admin_username,
            display_name=get_display_name(),
            metadata={"role": "admin", "provider": "credentials"},
        )


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the user session, set up preferences, and send a welcome message.
    """
    username, welcome_msg = await init_user_session()

    if not username or not welcome_msg:
        await cl.Message(content="Error initializing the chat. Please try again later.").send()
        return

    await cl.Message(
        content="",
        type="system_message",
        elements=[
            cl.CustomElement(
                **{
                    "name": "AssistantCard",
                    "props": {
                        "content": welcome_msg,
                        "author": "assistant",
                    },
                }
            )
        ],
    ).send()
    await get_chat_settings().send()

    res = await AskActionMessage(
        content="🤔 What vibe are you looking for today?",
        actions=[
            cl.Action(name="food", payload={"score": "food_score"}, label="🍕 I want amazing food!"),
            cl.Action(name="ambience", payload={"score": "ambience_score"}, label="🌟 The atmosphere"),
            cl.Action(name="price", payload={"score": "price_score"}, label="💸 Save money"),
            cl.Action(name="service", payload={"score": "service_score"}, label="👑 VIP service"),
        ],
        author="assistant",
        timeout=1200,
    ).send()

    if res and res.get("payload"):
        selected_score = res["payload"]["score"]

        await update_preferences(selected_score)
        await get_chat_settings().send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages from the user.
    This function checks if the user is authenticated, retrieves their preferences,
    and processes the message using the food bot agent.
    """
    # -- Agent processing message --
    agent = cl.user_session.get("agent")
    prefs = cl.user_session.get("user_preferences", {})
    params_chat = {"user_preferences": prefs}

    # -- If any next user response actions are set, remove them --
    await remove_next_response_actions()

    if prefs.get("distance_preference", False):
        params_chat.update({"distance_preference": True, "distance_km": prefs["distance_km"]})

    try:
        # -- Generate conversation summary and update chat title --
        cl.user_session.set("next_response_actions", [])
        thread_data = await cl_data._data_layer.get_thread(message.thread_id)
        message_history = [
            {"content": step["output"][:500], "role": "user" if step.get("type") == "user_message" else "assistant"}
            for step in thread_data["steps"][1:]  # Skip the first step which is the system message
            if step.get("type") in ["user_message", "assistant_message"]
        ]

        if len(message_history) > 0:
            print(f"Actual conversation: {message_history}")
            conversation_title = len(thread_data["name"].split())

            if conversation_title < 2 and len(message_history) > 1:
                conv_summary = await generate_conv_summary(message_history)
                chat_title = conv_summary.title().replace(".", "")
                await cl.context.emitter.init_thread(chat_title)

            next_response = await generate_next_response(message_history)
            next_response_actions = [
                cl.Action(
                    name="next_response",
                    payload={"user_response": response},
                    label=response,
                )
                for response in next_response.get("next_responses", [])
            ]

            cl.user_session.set("next_response_actions", next_response_actions)

        msg = cl.Message(content="", author="assistant", actions=cl.user_session.get("next_response_actions"))
        async_response = await cl.make_async(agent.params_chat)(message.content, **params_chat)
        for chunk in generate_streaming_response(async_response, 0.025):
            await msg.stream_token(chunk)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"Error processing message: {e}", author="assistant").send()


@cl.action_callback("next_response")
async def handle_next_response_action(action: cl.Action):
    """
    Handle the next response action when the user selects a response.
    This function updates the chat with the selected response and sends a follow-up message.
    """
    user_response = action.payload.get("user_response", "")
    user_message = cl.Message(content=user_response, author="user", type="user_message")

    await remove_next_response_actions()
    await user_message.send()
    await on_message(user_message)


@cl.on_settings_update
async def handle_settings_update(settings):
    """
    Only update the preferences dict in memory & in S3. Do NOT re-init the agent here.
    """
    try:
        config = cl.user_session.get("config")
        username: str = cl.user_session.get("username")
        prefs: dict = cl.user_session.get("user_preferences", {})
        s3_client = cl.user_session.get("s3_client")

        prefs.update(
            {
                "food_score": settings.get("food_score", prefs["food_score"]),
                "ambience_score": settings.get("ambience_score", prefs["ambience_score"]),
                "price_score": settings.get("price_score", prefs["price_score"]),
                "service_score": settings.get("service_score", prefs["service_score"]),
                "distance_preference": settings.get("distance_preference", prefs["distance_preference"]),
                "distance_km": settings.get("max_distance", prefs["distance_km"]),
            }
        )

        config["preferences"][username] = prefs
        s3_client.write_object(get_config_file(), yaml.dump(config))
        cl.user_session.set("user_preferences", prefs)

        await cl.Message(content="✅ Preferences updated!", author="assistant", type="system_message").send()
    except Exception as e:
        await cl.Message(content=f"Error saving preferences: {e}", author="assistant", type="system_message").send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume event"""
    if not thread or not cl.user_session.get("agent"):
        await init_user_session()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends"""
    pass  # no-op


@cl.step(type="tool", name="update_preferences")
async def update_preferences(selected_score: str):
    config = cl.user_session.get("config")
    username = cl.user_session.get("username")
    prefs: dict = cl.user_session.get("user_preferences", {})
    s3_client = cl.user_session.get("s3_client")

    messages = {
        "food_score": "Great! Let's find amazing food! 🍕 What's on your mind?",
        "ambience_score": "Perfect! I'll find places with great vibes! 🌟 What are you thinking?",
        "price_score": "Smart! I'll find budget-friendly options! 💸 What's your idea?",
        "service_score": "Excellent! I'll find places with top service! 👑 What can I help you with?",
    }

    all_scores = [value for key, value in prefs.items() if str(key).endswith("_score")]
    tuned_selected_score = max(all_scores) + 0.15

    prefs.update({selected_score: tuned_selected_score})
    prefs = normalize_weights(prefs)

    config["preferences"][username] = prefs
    s3_client.write_object(get_config_file(), yaml.dump(config))
    cl.user_session.set("user_preferences", prefs)

    await asyncio.sleep(1.0)
    return {key: value for key, value in prefs.items() if key in messages.keys()}
