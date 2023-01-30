import numpy as np

class buffer():

    def __init__(self, buffer_size: int):
        
        self.buffer_size = buffer_size
        self._buffer = []

    def push(self, frame:np.ndarray):
        if len(self._buffer) >= self.buffer_size:
            del self._buffer[0]
        self._buffer.append(frame)
      

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def is_full(self) -> bool:
        return len(self._buffer) == self.buffer_size
    
    def getter(self) -> np.ndarray: 
        return self._buffer

    def __str__(self) -> str:
        info = f'Total buffer size: {self.buffer_size} \n'\
               f'Now size: {len(self._buffer)}'
        return info