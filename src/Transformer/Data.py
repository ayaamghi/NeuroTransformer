#loads neural trace data for conditions, leads to some numpy arrays as output
from connectomics.common.utils import batch
from zapbench import constants
from zapbench import data_utils
from zapbench.ts_forecasting import data_source
import tensorstore as ts
import grain.python as grain


context = {
    "short": 4,
    "long": 256,
}
#get source for a split that can be iterated over to train model
class Split:
    def __init__(self, split: str, context_type: str, batch_size: int):
        self.batch_size = batch_size
        self.source = self.get_source(split, context_type)
        self.data_loader = self.get_data_loader()

    def get_data_loader(self):
        index_sampler = grain.IndexSampler(
    num_records=len(self.source),
    num_epochs=1,
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=True,
    seed=101
)
        return grain.DataLoader(
            data_source = self.source,
            sampler = index_sampler,
            operations= [
                grain.Batch(
                    batch_size=self.batch_size, drop_remainder=True
                )
            ],
            worker_count= 0
        )

    def get_source(self, split: str, context_type: str ):
        sources = []
        for condition_id in constants.CONDITIONS_TRAIN:
            config = data_source.TensorStoreTimeSeriesConfig(
                input_spec=data_utils.adjust_spec_for_condition_and_split(
                    condition=condition_id,
                    split=split,
                    spec=data_utils.get_spec('240930_traces'),
                    num_timesteps_context=4),
                timesteps_input=context[context_type],
                timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
            )
            sources.append(data_source.TensorStoreTimeSeries(config, prefetch=True))
        return data_source.ConcatenatedTensorStoreTimeSeries(*sources)


if __name__ == "__main__":
     train = Split("train", "short", 8)


