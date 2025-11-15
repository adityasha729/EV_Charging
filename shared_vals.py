
from multiprocessing import Manager

_manager = Manager()
gaze_state = _manager.dict({'page': 'unknown46'})
gaze_state_lock = _manager.Lock()
