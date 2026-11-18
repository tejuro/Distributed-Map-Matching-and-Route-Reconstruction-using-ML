
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap")
st.subheader("Distributed Map Matching & Route Reconstruction")
st.caption("GPS trajectory analysis and road-network route reconstruction")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input(
        "Backend API URL",
        value="http://127.0.0.1:8000"
    )
    trip_id = st.selectbox(
        "Select Evaluation Trip",
        [f"Trip {i}" for i in range(1, 11)]
    )

st.header("Route Reconstruction")

if st.button("Reconstruct Route", type="primary"):
    with st.spinner("Contacting the backend..."):
        try:
            # Temporary API contract; coordinate with Member 3.
            response = requests.post(
                f"{api_url.rstrip('/')}/predict",
                json={"trip_id": trip_id},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            st.session_state["result"] = result
            st.session_state["selected_trip"] = trip_id

        except requests.RequestException as error:
            st.error(f"Backend connection failed: {error}")
            st.info("The backend must be running to generate predictions.")

result = st.session_state.get("result")

if result:
    st.success(
        f"Prediction received for "
        f"{st.session_state.get('selected_trip', trip_id)}"
    )

    metrics = result.get("metrics", {})
    if metrics:
        st.subheader("Evaluation Metrics")
        columns = st.columns(min(3, len(metrics)))
        for index, (name, value) in enumerate(metrics.items()):
            columns[index % len(columns)].metric(
                name.replace("_", " ").title(),
                str(value)
            )

    st.subheader("Predicted Route")

    # Expected format: [[latitude, longitude], ...]
    route = result.get("predicted_route", [])

    if route:
        route_map = folium.Map(
            location=route[0],
            zoom_start=14
        )
        folium.PolyLine(
            route,
            color="blue",
            weight=5,
            tooltip="Predicted Route"
        ).add_to(route_map)

        folium.Marker(
            route[0], tooltip="Start"
        ).add_to(route_map)
        folium.Marker(
            route[-1], tooltip="End"
        ).add_to(route_map)

        st_folium(route_map, width=None, height=500)
    else:
        st.warning("No route coordinates returned by the backend.")

    with st.expander("View Backend Response"):
        st.json(result)

else:
    st.info(
        "Select an evaluation trip and click "
        "'Reconstruct Route' to request a prediction."
    )
