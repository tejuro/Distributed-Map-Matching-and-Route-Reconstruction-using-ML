from fastapi import FastAPI, HTTPException

from backend.services import get_available_trips
from backend.route_service import predict_route

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map Matching and Route Reconstruction",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "PathSnap Backend"}

@app.get("/trips")
def list_trips():
    return {"trips": get_available_trips()}

@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str):
    if trip_id not in get_available_trips():
        raise HTTPException(
            status_code=404,
            detail=f"Trip {trip_id} not found"
        )

    try:
        return predict_route(trip_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Route prediction failed"
        ) from exc
