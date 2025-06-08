import os

import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    dotenv.load_dotenv(dotenv_path=dotenv_path)

import streamlit as st
import streamlit_authenticator as st_auth
import streamlit_geolocation as st_geolocation
import yaml

from src.helper.utils import generate_streaming_response, get_welcome_message

# --- Variables setup ---
assistant_avatar = "public/favicon.png"
user_avatar = "👤"
CONFIG_FILE = "secret.yaml"


def streamlit_main(page_title: str = "TripAdvisor Chatbot"):
    """
    Main function to run the Streamlit application for the TripAdvisor Chatbot.
    This function initializes the Streamlit app, sets up the S3 client, loads the configuration,
    initializes the agent, and handles user authentication and preferences.
    Args:
        page_title (str): Title of the Streamlit page.
    """
    # --- Set page configuration ---
    st.set_page_config(page_title=page_title, page_icon="public/favicon.png", layout="centered")

    # --- Initialize S3 client ---
    if "s3_client" not in st.session_state and not st.session_state.get("skip_agent_init"):
        try:
            from src.main import s3_client

            st.session_state.s3_client = s3_client
        except Exception as e:
            st.error(f"Error initializing S3 client: {str(e)}")
            st.stop()

    # --- Initialize config file in S3 if it doesn't exist ---
    try:
        # Check if secret.yaml exists in S3
        objects = st.session_state.s3_client.list_objects()
        if not any(obj["file"] == CONFIG_FILE for obj in objects):
            default_config = {
                "credentials": {"usernames": {}},
                "preferences": {},
            }
            # Write default config to S3
            st.session_state.s3_client.write_object(CONFIG_FILE, yaml.dump(default_config, default_flow_style=False))
    except Exception as e:
        st.error(f"Error initializing config file in S3: {str(e)}")
        st.stop()

    # --- Load config file from S3 ---
    try:
        config_data = st.session_state.s3_client.read_object(CONFIG_FILE)
        config = yaml.safe_load(config_data)
    except Exception as e:
        st.error(f"Error loading config file from S3: {str(e)}")
        st.stop()

    # --- Initialize agent ---
    if not st.session_state.get("agent") and not st.session_state.get("skip_agent_init"):
        try:
            from src.main import generate_foodbot_agent

            st.session_state.skip_agent_init = True  # prevent re-initialization
            st.session_state.agent = generate_foodbot_agent(
                chat_store_token_limit=4096, verbose=True, callback="streamlit"
            )
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.stop()

    # --- Initialize authenticator ---
    try:
        authenticator = st_auth.Authenticate(credentials=config["credentials"])
    except Exception as e:
        st.error(f"Error initializing authenticator: {str(e)}")
        st.stop()

    # --- Initialize variables to avoid NameError ---
    default_preferences = {
        "food_score": 0.55,
        "ambience_score": 0.15,
        "price_score": 0.15,
        "service_score": 0.15,
        "distance_preference": False,
        "distance_km": 15,
    }

    # --- Sidebar for login or sign-up ---
    with st.sidebar:
        gps_location = st_geolocation.streamlit_geolocation()
        option = st.selectbox("Choose an option", ["Login", "Sign Up"])

        if option == "Login":
            authenticator.login(key="Login", location="sidebar")
            st.sidebar.write("Auth Status:", st.session_state.authentication_status)

            if not st.session_state.authentication_status:
                st.warning("Please enter your username and password to login.")

            if "preferences" not in config:
                config["preferences"] = {}
            if st.session_state.get("username") not in config["preferences"]:
                config["preferences"][st.session_state.get("username")] = default_preferences
                try:
                    st.session_state.s3_client.write_object(CONFIG_FILE, yaml.dump(config, default_flow_style=False))
                except Exception as e:
                    st.error(f"Error saving default preferences to S3: {str(e)}")
                    st.stop()

        elif option == "Sign Up":
            st.markdown("### Sign Up")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Sign Up"):
                try:
                    config_data = st.session_state.s3_client.read_object(CONFIG_FILE)
                    config = yaml.safe_load(config_data)
                    if username in config["credentials"]["usernames"]:
                        st.error("Username already exists")
                    elif not username or not password:
                        st.error("All fields are required")
                    else:
                        hashed_password = st_auth.Hasher.hash(password)
                        config["credentials"]["usernames"][username] = {"name": username, "password": hashed_password}
                        st.session_state.s3_client.write_object(
                            CONFIG_FILE, yaml.dump(config, default_flow_style=False)
                        )
                        st.success("Registration successful! Please select 'Login' to continue.")
                except Exception as e:
                    st.error(f"Error during sign-up: {str(e)}")

        if st.button("🧹 Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session state cleared.")
            st.rerun()

    if option == "Login" and st.session_state.authentication_status:
        # --- Initialize session state with user preferences ---
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = config["preferences"][st.session_state.username]

        # --- Sidebar for preferences and logout ---
        if "username" in st.session_state:
            st.sidebar.info(f"👤 Welcome, {st.session_state.get('username')}")
            authenticator.logout("Logout", "sidebar")

            # --- Welcome message and preferences setup ---
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(get_welcome_message(name=st.session_state.get("name")), unsafe_allow_html=True)

                st.session_state.distance_preference = st.checkbox(
                    "Prefer nearby restaurants",
                    value=st.session_state.user_preferences.get("distance_preference", False),
                )
                if st.session_state.distance_preference:
                    st.session_state.distance_km = st.slider(
                        "Maximum distance (km)",
                        1,
                        30,
                        value=st.session_state.user_preferences.get("distance_km", 15),
                        step=1,
                    )
                col1, col2 = st.columns(2)
                with col1:
                    food_score = st.slider(
                        "🍽️ Food",
                        0.0,
                        1.0,
                        value=st.session_state.user_preferences.get("food_score", 0.55),
                        step=0.05,
                    )
                    price_score = st.slider(
                        "💰 Price",
                        0.0,
                        1.0,
                        value=st.session_state.user_preferences.get("price_score", 0.15),
                        step=0.05,
                    )
                with col2:
                    ambience_score = st.slider(
                        "🏢 Ambience",
                        0.0,
                        1.0,
                        value=st.session_state.user_preferences.get("ambience_score", 0.15),
                        step=0.05,
                    )
                    service_score = st.slider(
                        "👨‍🍳 Service",
                        0.0,
                        1.0,
                        value=st.session_state.user_preferences.get("service_score", 0.15),
                        step=0.05,
                    )

                if st.button("Save Preferences"):
                    new_preferences = {
                        "food_score": food_score,
                        "ambience_score": ambience_score,
                        "price_score": price_score,
                        "service_score": service_score,
                    }
                    try:
                        if st.session_state.username:
                            config_data = st.session_state.s3_client.read_object(CONFIG_FILE)
                            config = yaml.safe_load(config_data)
                            config["preferences"][st.session_state.username] = new_preferences
                            st.session_state.s3_client.write_object(
                                CONFIG_FILE, yaml.dump(config, default_flow_style=False)
                            )
                            st.session_state.user_preferences = new_preferences
                            st.success("Preferences saved!")
                    except Exception as e:
                        st.error(f"Error saving preferences to S3: {str(e)}")

        # --- Display chat history ---
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
            ):
                st.markdown(message["content"])

        # --- Handle user input ---
        if prompt := st.chat_input("Planning your next meal? Ask me anything!"):
            if "progress_bar" in st.session_state:
                st.session_state.progress_bar.empty()
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=user_avatar):
                st.markdown(prompt)

            st.session_state.status = st.status("Thinking...")
            st.session_state.progress_bar = st.progress(0)
            try:
                params_chat = {"user_preferences": st.session_state.user_preferences}
                if st.session_state.distance_preference:
                    if gps_location and gps_location.get("latitude") and gps_location.get("longitude"):
                        params_chat["user_preferences"]["distance_preference"] = True
                        params_chat["user_preferences"]["distance_km"] = st.session_state.distance_km
                        params_chat["user_preferences"]["user_lat"] = gps_location["latitude"]
                        params_chat["user_preferences"]["user_long"] = gps_location["longitude"]
                    else:
                        st.toast("Please click on the button on the left to allow location access", icon="⚠️")
                response = st.session_state.agent.params_chat(prompt, **params_chat)
                st.session_state.progress_bar.progress(100)
                st.toast("Agent has ranked the restaurants and found the best for you", icon="✅")
                st.session_state.messages.append({"role": "assistant", "content": str(response)})

                with st.chat_message("assistant", avatar=assistant_avatar):
                    st.write_stream(generate_streaming_response(str(response)))
            except Exception as e:
                st.error(f"Error processing chat: {str(e)}")


if __name__ == "__main__":
    streamlit_main(page_title="Food Advisor")
