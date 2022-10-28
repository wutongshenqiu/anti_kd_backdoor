from typing import Iterable

from gpustat import GPUStatCollection


class GpuHelper:

    def __init__(self,
                 *,
                 memory_threshold: float = 1e4,
                 utilization_threshold: float = 0.5,
                 last_seconds: int = 10) -> None:
        self._memory_threshold = memory_threshold
        self._utilization_threshold = utilization_threshold
        self._last_seconds = last_seconds

    def get_available_indices(self) -> list[int]:
        available_indices = set(range(len(self.gpu_stats)))

        for _ in range(self._last_seconds):
            cur_available_indices = {
                s['index']
                for s in self.gpu_stats
                if s['utilization.gpu'] < self._utilization_threshold * 100 and
                s['memory.total'] - s['memory.used'] > self._memory_threshold
            }

            available_indices &= cur_available_indices

        return list(available_indices)

    def wait_for_available_indices(self, gpu_number: int = 1) -> list[int]:
        while len((available_indices := self.get_available_indices())) < \
                gpu_number:
            ...

        return available_indices[:gpu_number]

    @property
    def gpu_stats(self) -> list[dict]:
        return GPUStatCollection.new_query().jsonify()['gpus']

    def set_visiable_devices(self, indices: Iterable[int]) -> None:
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, indices))


if __name__ == '__main__':
    gpu_helper = GpuHelper()
    print(gpu_helper.wait_for_available_indices(2))
