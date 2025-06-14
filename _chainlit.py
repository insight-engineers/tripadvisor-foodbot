import chainlit as cl
import yaml

from src.helper.utils import (
    encode_b64_string,
    generate_streaming_response,
    get_admin_account,
    get_config_file,
)
from src.main import (
    get_chat_settings,
    init_user_session,
    s3_client,
)


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    session_username, session_password = encode_b64_string(username), encode_b64_string(password)
    admin_name_display, admin_username, admin_password = get_admin_account()

    if (session_username, session_password) == (admin_username, admin_password):
        return cl.User(
            identifier=admin_username,
            display_name=admin_name_display,
            metadata={"role": "admin", "provider": "credentials"},
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the user session, set up preferences, and send a welcome message.
    """
    username, welcome_msg = await init_user_session()

    if not username or not welcome_msg:
        await cl.Message(content="Error initializing the chat. Please try again later.").send()
        return

    await cl.Message(content=welcome_msg, author="assistant").send()
    await get_chat_settings().send()

    res = await cl.AskActionMessage(
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

        config = cl.user_session.get("config")
        username = cl.user_session.get("username")
        prefs = cl.user_session.get("user_preferences", {})

        for score in ["food_score", "ambience_score", "price_score", "service_score"]:
            prefs[score] = 0.5
        prefs[selected_score] = 1.0

        config["preferences"][username] = prefs
        s3_client.write_object(get_config_file(), yaml.dump(config))
        cl.user_session.set("user_preferences", prefs)

        await get_chat_settings().send()


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming messages from the user.
    This function checks if the user is authenticated, retrieves their preferences,
    and processes the message using the food bot agent.
    """
    agent = cl.user_session.get("agent")
    prefs = cl.user_session.get("user_preferences", {})
    params_chat = {"user_preferences": prefs}

    if prefs.get("distance_preference", False):
        params_chat.update({"distance_preference": True, "distance_km": prefs["distance_km"]})

    try:
        msg: cl.Message = cl.Message(content="", author="Assistant")
        async_response = await cl.make_async(agent.params_chat)(message.content, **params_chat)

        for chunk in generate_streaming_response(async_response):
            await msg.stream_token(chunk)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"Error processing message: {e}", author="assistant").send()


@cl.on_settings_update
async def handle_settings_update(settings):
    """
    Only update the preferences dict in memory & in S3. Do NOT re-init the agent here.
    """
    # Check authentication
    user = cl.user_session.get("user")
    if not user or user.identifier == "anonymous":
        await cl.Message(content="Please authenticate to update preferences.").send()
        return

    try:
        config = cl.user_session.get("config")
        username = cl.user_session.get("username")
        prefs = cl.user_session.get("user_preferences", {})

        # Overwrite only fields that came from the UI; keep others untouched
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

        # Persist back to S3
        config["preferences"][username] = prefs
        s3_client.write_object(get_config_file(), yaml.dump(config))
        cl.user_session.set("user_preferences", prefs)

        await cl.Message(content="✅ Preferences updated!", author="assistant").send()
    except Exception as e:
        await cl.Message(content=f"Error saving preferences: {e}", author="assistant").send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume event"""
    pass


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends"""
    pass
