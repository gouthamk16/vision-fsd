from detect import VehicleTracker
from extract import FeatureExtractor
    
def toggle_vehicle_only(self):
    return VehicleTracker.toggle_vehicle_only()

def get_detection_display(self):
    return VehicleTracker.create_detection_display()

def get_trajectory_display(self):
    return FeatureExtractor.get_trajectory_display()