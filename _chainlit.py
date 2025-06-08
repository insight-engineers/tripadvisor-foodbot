import chainlit as cl
import chainlit.input_widget as cliw
import yaml

from src.helper.utils import generate_streaming_response, get_welcome_message
from src.main import generate_foodbot_agent, s3_client

CONFIG_FILE = "secret.yaml"


# OAuth provider configuration
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> cl.User:
    """
    Handle OAuth callback and create user from GitHub data
    """
    if provider_id == "github":
        # Extract GitHub username and other info
        github_username = raw_user_data.get("login")
        github_name = raw_user_data.get("name", github_username)
        github_email = raw_user_data.get("email")

        return cl.User(
            identifier=github_username,  # Use GitHub username as identifier
            metadata={
                "name": github_name,
                "email": github_email,
                "avatar_url": f"https://github.com/{github_username}.png",
                "provider": "github",
                "raw_data": raw_user_data,
            },
        )

    return None


# Add WebSocket connection handler to prevent auth errors
@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends"""
    pass


async def init_user_session():
    """
    1. Get authenticated user
    2. Initialize S3 client
    3. Load (or create) CONFIG_FILE on S3
    4. Ensure user preferences exist
    5. Create the agent and store everything in user_session
    Returns: username, welcome message
    """
    # -- 1. Get authenticated user
    user = cl.user_session.get("user")
    if not user or user.identifier == "anonymous":
        await cl.Message(content="Authentication required. Please log in with GitHub.").send()
        return None, None

    username = user.identifier  # This is the GitHub username

    # -- 2. Load/initialize config file on S3
    try:
        objects = s3_client.list_objects()
        if not any(obj["file"] == CONFIG_FILE for obj in objects):
            default_conf = {"credentials": {"usernames": {}}, "preferences": {}}
            s3_client.write_object(CONFIG_FILE, yaml.dump(default_conf))

        raw = s3_client.read_object(CONFIG_FILE)
        config = yaml.safe_load(raw)
        cl.user_session.set("config", config)
    except Exception as e:
        await cl.Message(content=f"Error handling config file in S3: {e}").send()
        return None, None

    # -- 3. Set user and preferences
    cl.user_session.set("username", username)

    default_prefs = {
        "food_score": 0.55,
        "ambience_score": 0.15,
        "price_score": 0.15,
        "service_score": 0.15,
        "distance_preference": False,
        "distance_km": 15,
        "user_lat": 0.0,
        "user_long": 0.0,
    }
    prefs_container = config.setdefault("preferences", {})
    user_prefs = prefs_container.get(username, default_prefs.copy())

    if username not in prefs_container:
        prefs_container[username] = user_prefs
        s3_client.write_object(CONFIG_FILE, yaml.dump(config))

    cl.user_session.set("user_preferences", user_prefs)

    # -- 4. Agent instantiation
    try:
        agent = generate_foodbot_agent(chat_store_token_limit=4096, verbose=True, callback="chainlit")
        cl.user_session.set("agent", agent)
    except Exception as e:
        await cl.Message(content=f"Error initializing agent: {e}").send()
        return None, None

    # -- 5. Welcome message with user's name
    display_name = user.metadata.get("name", username)
    wm = get_welcome_message(name=display_name)
    return username, wm


@cl.on_chat_start
async def start():
    # Check if user is properly authenticated
    user = cl.user_session.get("user")
    if not user or user.identifier == "anonymous":
        await cl.Message(content="Please authenticate with GitHub to access the food bot.").send()
        return

    username, welcome_msg = await init_user_session()

    if not username or not welcome_msg:
        await cl.Message(content="Error initializing the chat. Please try again later.").send()
        return

    await cl.Message(content=welcome_msg, author="assistant").send()

    # Load user's existing preferences for the sliders
    user_prefs = cl.user_session.get("user_preferences", {})

    await cl.ChatSettings(
        [
            cliw.Slider(
                id="food_score",
                label="Food Score",
                min=0.0,
                max=1.0,
                step=0.05,
                initial=user_prefs.get("food_score", 0.55),
            ),
            cliw.Slider(
                id="ambience_score",
                label="Ambience Score",
                min=0.0,
                max=1.0,
                step=0.05,
                initial=user_prefs.get("ambience_score", 0.15),
            ),
            cliw.Slider(
                id="price_score",
                label="Price Score",
                min=0.0,
                max=1.0,
                step=0.05,
                initial=user_prefs.get("price_score", 0.15),
            ),
            cliw.Slider(
                id="service_score",
                label="Service Score",
                min=0.0,
                max=1.0,
                step=0.05,
                initial=user_prefs.get("service_score", 0.15),
            ),
            cliw.Switch(
                id="distance_preference",
                label="Prefer Nearby Restaurants",
                initial=user_prefs.get("distance_preference", False),
            ),
            cliw.Slider(
                id="max_distance",
                label="Max Distance (km)",
                min=1,
                max=30,
                step=1,
                initial=user_prefs.get("distance_km", 15),
            ),
            cliw.NumberInput(id="user_lat", label="Latitude", initial=user_prefs.get("user_lat", 0.0)),
            cliw.NumberInput(id="user_long", label="Longitude", initial=user_prefs.get("user_long", 0.0)),
        ]
    ).send()


@cl.action_callback("increase_food_score")
async def increase_food_score():
    """Increase food score preference"""
    prefs = cl.user_session.get("user_preferences", {})
    prefs["food_score"] = min(prefs.get("food_score", 0.55) + 0.05, 1.0)
    cl.user_session.set("user_preferences", prefs)
    await cl.Message(content="Food score preference increased!").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # Check authentication
    user = cl.user_session.get("user")
    if not user or user.identifier == "anonymous":
        await cl.Message(content="Please authenticate with GitHub to use the food bot.").send()
        return

    agent = cl.user_session.get("agent")
    prefs = cl.user_session.get("user_preferences", {})
    params_chat = {"user_preferences": prefs}

    # If distance preference is on, but lat/long are zero, prompt to set them
    if prefs.get("distance_preference", False):
        lat, lon = prefs.get("user_lat", 0.0), prefs.get("user_long", 0.0)
        if lat == 0.0 or lon == 0.0:
            await cl.Message(
                content="Please set valid Latitude and Longitude in chat settings for distance-based search.",
                author="assistant",
            ).send()
            return
        params_chat.update(
            {
                "distance_preference": True,
                "distance_km": prefs["distance_km"],
                "user_lat": lat,
                "user_long": lon,
            }
        )

    try:
        msg: cl.Message = cl.Message(content="", author="Assistant")
        async_response = await cl.make_async(agent.params_chat)(message.content, **params_chat)

        for chunk in generate_streaming_response(async_response):
            await msg.stream_token(chunk)

        await msg.update()

        if len(str(async_response)) > 512:  # Ask for feedback if the response is detailed
            await cl.Message(
                content="Is the recommendation helpful? If not, please provide more details.",
                author="assistant",
                actions=[
                    cl.Action(name="helpful_action", label="✅ Yes, it was helpful", payload={"helpful": True}),
                    cl.Action(name="helpful_action", label="❌ No, I need something else", payload={"helpful": False}),
                ],
            ).send()

    except Exception as e:
        await cl.Message(content=f"Error processing message: {e}", author="assistant").send()


@cl.action_callback("helpful_action")
async def handle_helpful_action(action):
    """
    Handle the user's feedback on the recommendation.
    If helpful, thank them. If not, ask for more details.
    """
    if action.payload.get("helpful"):
        await cl.Message(content="Thank you! I'm glad the recommendation was helpful!").send()
    else:
        await cl.Message(
            content="Sorry to hear that! Please provide more details about what you're looking for."
        ).send()

    await action.remove()


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
                "user_lat": settings.get("user_lat", prefs["user_lat"]),
                "user_long": settings.get("user_long", prefs["user_long"]),
            }
        )

        # Persist back to S3
        config["preferences"][username] = prefs
        s3_client.write_object(CONFIG_FILE, yaml.dump(config))
        cl.user_session.set("user_preferences", prefs)

        await cl.Message(content="✅ Preferences updated!", author="assistant").send()
    except Exception as e:
        await cl.Message(content=f"Error saving preferences: {e}", author="assistant").send()
