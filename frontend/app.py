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
st.caption("GPS trajectory analysis using XGBoost and HMM/Viterbi")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input(
        "Backend API URL",
        value="http://127.0.0.1:8000"
    ).rstrip("/")

    try:
        response = requests.get(f"{api_url}/trips", timeout=10)
        response.raise_for_status()
        trips = response.json()["trips"]
    except requests.RequestException:
        st.error("Backend is not connected.")
        trips = []

    if not trips:
        st.stop()

    trip_id = st.selectbox("Select Evaluation Trip", trips)

st.header("Route Reconstruction")

if st.button("Reconstruct Route", type="primary"):
    try:
        with st.spinner("Reconstructing route..."):
            response = requests.post(
                f"{api_url}/predict/{trip_id}",
                timeout=120
            )
            response.raise_for_status()
            st.session_state["result"] = response.json()

    except requests.RequestException as error:
        st.error(f"Prediction failed: {error}")

result = st.session_state.get("result")

if result and result.get("trip_id") == trip_id:
    st.success(f"Route reconstructed for trip {trip_id}")

    metrics = result["metrics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("GPS Observations", metrics["gps_observations"])
    c2.metric("Matched Roads", metrics["road_segments"])
    c3.metric("HMM Restarts", metrics["hmm_restarts"])

    gps = result["gps_points"]
    roads = result["predicted_route"]

    st.subheader("GPS vs Predicted Route")

    if gps:
        route_map = folium.Map(
            location=gps[0],
            zoom_start=14
        )

        folium.PolyLine(
            gps,
            color="blue",
            weight=2,
            opacity=0.6,
            tooltip="Original GPS Track"
        ).add_to(route_map)

        for road in roads:
            folium.PolyLine(
                [road["start"], road["end"]],
                color="red",
                weight=4,
                opacity=0.8
            ).add_to(route_map)

        folium.Marker(
            gps[0],
            tooltip="Start",
            icon=folium.Icon(color="green")
        ).add_to(route_map)

        folium.Marker(
            gps[-1],
            tooltip="End",
            icon=folium.Icon(color="red")
        ).add_to(route_map)

        st_folium(route_map, width=None, height=550)

        st.caption(
            "Blue = Original GPS | Red = Predicted Road Segments. "
            "HMM restarts may cause gaps in the matched route."
        )

    with st.expander("View Backend Response"):
        st.json(result)

else:
    st.info("Select a trip and click Reconstruct Route.")
