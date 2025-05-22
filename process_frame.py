# Logic to calculate the distance between the car in the pov and the surrounding cars.

from detect import VehicleTracker

class Processor:
    def __init__(self, frame):
        self.raw_frame = frame # raw frame - unprocessed frame straight from the video
        self.processed_frame = None

    def calculate_distance(self):
        tracker = VehicleTracker()
        bb_coords = tracker.track(frame=self.raw_frame)
        self.processed_frame = tracker.draw_bb(frame=self.raw_frame, bounding_box_coords=bb_coords)
        return self.processed_frame
