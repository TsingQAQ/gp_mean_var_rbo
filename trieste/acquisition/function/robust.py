from __future__ import annotations

from abc import ABC
from functools import partial
from itertools import combinations, product
from typing import Callable, Mapping, Optional, Tuple, cast

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from trieste.utils.wrapper import sequential_batch

from ...data import Dataset
from ...models import ProbabilisticModel
from ...space import DiscreteSearchSpace, SearchSpace
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder,
    T,
)
from ..multi_objective.pareto import Pareto, get_reference_point, non_dominated
from ..multi_objective.partition import prepare_default_non_dominated_partition_bounds
from ..optimizer import generate_continuous_optimizer
from ..sampler import QFFVarianceSampler, RFFVarianceSampler


def fmean_mean(input, model, noisy_disy, sample_number):
    input_noisy_dists = noisy_disy.sample(sample_number)
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        input_noisy_dists, dtype=input.dtype
    )  # [N, mc, dim]
    return tf.reduce_mean(tf.squeeze(model.predict(input_dists)[0], axis=-1), -1, keepdims=True)


def fvar_mean(input, model, noisy_disy, sample_number):
    input_noisy_dists = noisy_disy.sample(sample_number)
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        input_noisy_dists, dtype=input.dtype
    )  # [N, mc, dim]
    return tf.math.reduce_variance(
        tf.squeeze(model.predict(input_dists)[0], axis=-1), -1, keepdims=True
    )


class FF_MV_qEHVI(SingleModelAcquisitionBuilder):
    def __init__(
        self,
        noise_type,
        noise_param: [list, TensorType],
        ff_method: str = "RFF",
        opt_ff_num: int = 1000,
        infer_mc_num: int = 10000,
        mc_num: int = 32,
        max_batch_element: int = 500,
        implicit_sample: bool = False,
        extract_ref_point_from_mean: bool = True
    ):
        """
        :param noise_type
        :param noise_param: representing variance in case noise_type is normal
        :param ff_method
        :param opt_ff_num
        :param mc_num
        """
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._ff_method = ff_method
        self._opt_ff_num = opt_ff_num
        self._infer_mc_num = infer_mc_num
        self._mc_num = mc_num
        self._noise_type = noise_type
        self._noise_param = noise_param
        self.fmean_mean = None
        self.fvar_mean = None
        self._max_batch_element = max_batch_element
        self._implicit_sample = implicit_sample
        self._ref_point_use_mean = extract_ref_point_from_mean

    def prepare_acquisition_function(
        self,
        model: Mapping[str:ProbabilisticModel],
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:

        def get_feasible_fmin_samples_from_traj(sampled_trajectories: list[Callable, Callable]):
            mean_traj, var_traj = sampled_trajectories
            mean_trajs = mean_traj(dataset.query_points)  # [mc, N, 1]
            var_trajs = var_traj(dataset.query_points)  # [mc, N, 1]
            partition_lbs = []
            partition_ubs = []
            __ref_pts = []
            __mean_var_trajs = []
            # first loop: determin the ref point
            for mean_traj, var_traj in zip(mean_trajs, var_trajs):
                mean_var_traj = tf.concat([mean_traj, var_traj], axis=-1)
                __mean_var_trajs.append(mean_var_traj)
                __ref_pts.append(tf.expand_dims(get_reference_point(Pareto(mean_var_traj).front), axis=0))
            if not self._ref_point_use_mean:
                ref_pt = tf.reduce_max(tf.concat(__ref_pts, axis=0), axis=0)
            else:
                _inferred_obs = tf.concat(
                    [self.fmean_mean(dataset.query_points), self.fvar_mean(dataset.query_points)], -1
                )
                ref_pt = get_reference_point(Pareto(_inferred_obs).front)

            for _mean_var_traj in __mean_var_trajs:
                _screened_front = Pareto(_mean_var_traj).front[tf.reduce_all(Pareto(_mean_var_traj).front <= ref_pt, -1)]
                __lb, __ub = prepare_default_non_dominated_partition_bounds(
                    ref_pt, _screened_front
                )
                partition_lbs.append(__lb.numpy())
                partition_ubs.append(__ub.numpy())
            # note: each -2 axis's length can be different
            return (tf.ragged.constant(partition_lbs, ragged_rank=1),
                    tf.ragged.constant(partition_ubs, ragged_rank=1))

        # construct partition bounds
        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )

        # # construct sampled trajectory
        if self._ff_method == "RFF":
            sampler = RFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )
        else:
            assert self._ff_method == "QFF"
            sampler = QFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )
        fmean, theta_sample = sampler.get_mean_trajectory(
            sample_size=self._mc_num,
            return_sample=True,
            max_batch_element_num=self._max_batch_element,
        )
        var = sampler.get_var_trajectory(
            theta_sample=theta_sample, max_batch_element_num=self._max_batch_element
        )
        trajectories = [fmean, var]

        # get partitioned bounds
        if self._implicit_sample is False:
            inferred_obs = tf.concat(
                [self.fmean_mean(dataset.query_points), self.fvar_mean(dataset.query_points)], -1
            )
            _partition_bounds = prepare_default_non_dominated_partition_bounds(
                get_reference_point(Pareto(inferred_obs).front), Pareto(inferred_obs).front
            )
            # expand shape
            _lbs = tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(_partition_bounds[0], axis=0), self._mc_num, axis=0))
            _ubs = tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(_partition_bounds[1], axis=0), self._mc_num, axis=0))
            _partition_bounds = (_lbs, _ubs)

        else:
            _partition_bounds = get_feasible_fmin_samples_from_traj(trajectories)
        return batch_mva_ehvi(trajectories, _partition_bounds)


def batch_mva_ehvi(
    sampled_trajectories: list,
    partition_bounds: tuple[TensorType, TensorType],
    pending_points: Optional[TensorType] = None,
) -> AcquisitionFunction:
    """
    :param sampled_trajectories: The posterior sampler, which given query points `at`, is able to sample
        the possible observations at 'at'.
    :param partition_bounds: with shape ([1, N, D], [1, N, D]), or with shape ([mc_num, None, D], [mc_num, None, D])
        partitioned non-dominated hypercell bounds for hypervolume improvement calculation.
    :param pending_points: Points already chosen to be in the current batch (of shape [M,D]).
    :return: The batch expected hypervolume improvement acquisition
        function for objective minimisation.
    """

    def acquisition(at: TensorType) -> TensorType:
        # _batch_size = 1 if pending_points is None else pending_points.shape[0] + 1  # B
        _batch_size = at.shape[-2]

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return tf.ragged.constant([list(combinations(indices, i)) for i in range(1, q + 1)])

        if pending_points is not None:
            raise NotImplementedError
        else:  # pending point is none, we squeeze at: assume it can only be sequential here
            # assert tf.shape(at)[-2] == 1, ValueError('FF-MV-qEHVI only support greedy batch')
            mean_traj, var_traj = sampled_trajectories
            # at [N ,B, dim] -> [N * B, dim] # implicitly assume at with rank 3
            _hack_at = tf.reshape(at, (tf.shape(at)[0] * tf.shape(at)[1], tf.shape(at)[-1]))
            res = tf.concat([mean_traj(_hack_at), var_traj(_hack_at)], -1)  # [_mc_num, N * B, 2]
            res = tf.reshape(
                res, (tf.shape(res)[0], tf.shape(at)[0], tf.shape(at)[1], tf.shape(res)[-1])
            )  # [_mc_num, N, B, 2]
            samples = tf.transpose(res, perm=[1, 0, 2, 3])  # [N, _mc_num, B, 2]

        q_subset_indices = gen_q_subset_indices(_batch_size)

        lb_points, ub_points = partition_bounds

        def hv_contrib_on_samples(
            obj_samples: TensorType,
        ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj], e.g., [batch_size, mc_size, Cq_j, num_obj]
            overlap_vertices = tf.reduce_max(obj_samples, axis=-2)
            # [batch_size, mc_size, cell_size, Cq_j, num_obj] -> [batch_size, mc_size, cell_size, Cq_j, num_obj]
            # Note: Raggered Tensor has some unideal behaviors regarding broadcasting, hence the following code
            # is a bit weird written, may need improve in future!
            overlap_vertices = tf.maximum(  # compare overlap vertices and lower bound of each cell:
                tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                lb_points[:, :, tf.newaxis, :],
            )  # [..., S, K, Cq_j, num_obj]

            lengths_j = tf.maximum(  # get hvi length per obj within each cell
                (ub_points[tf.newaxis, :, :, tf.newaxis, :] - overlap_vertices), 0.0
            )  # [..., S, K, Cq_j, num_obj]

            areas_j = tf.reduce_sum(  # sum over all subsets Cq_j -> [..., S, K]
                tf.reduce_prod(lengths_j, axis=-1), axis=-1  # calc hvi within each K
            )

            return tf.reduce_sum(areas_j, axis=-1)  # sum over cells -> [..., S]

        assert _batch_size == 1
        # TODO: Force use the following (don't consider IEP), future use CBD
        q_choose_j = tf.gather(q_subset_indices, 0).to_tensor()
        return tf.reduce_mean(hv_contrib_on_samples(tf.gather(samples, q_choose_j, axis=-2)), axis=-1, keepdims=True)

    return acquisition


class Random_Acq(SingleModelAcquisitionBuilder):
    def __init__(
        self,
        noise_type,
        noise_param: [list, TensorType],
        infer_mc_num: int = 10000,
    ):
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            # TODO: Test of uniform
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._infer_mc_num = infer_mc_num
        self._noise_type = noise_type
        self._noise_param = noise_param
        self.fmean_mean = None
        self.fvar_mean = None

    def prepare_acquisition_function(
        self,
        model: Mapping[str:ProbabilisticModel],
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:

        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )


class US(SingleModelAcquisitionBuilder):
    """
    Uncertainty Sampling, used in iwazaki2021mean
    US chooses xt such that it achieves the largest average posterior variance

    xt = argmax_x∈X \int σt−1(x,w) p(w) dw
    """

    def __init__(
        self,
        noise_type,
        noise_param: [list, TensorType],
        infer_mc_num: int = 10000,
    ):
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            # TODO: Test of uniform
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._infer_mc_num = infer_mc_num
        self.fmean_mean = None
        self.fvar_mean = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        noisy_samples = self._noise_dist.sample(self._infer_mc_num)
        return uncertainty_sampling(noisy_samples, model)


def uncertainty_sampling(
    noisy_samples: TensorType,
    model: ProbabilisticModel,
    pending_points: Optional[TensorType] = None,
) -> AcquisitionFunction:
    # @tf.function
    def acquisition(at: TensorType) -> TensorType:
        at = tf.squeeze(at, -2)
        input_dists = tf.expand_dims(at, -2) + tf.cast(noisy_samples, dtype=at.dtype)  # [N, mc, 1]
        n_means, n_vars = model.predict(input_dists)  # [N, mc, 1]
        return tf.reduce_mean(tf.sqrt(n_vars), -2)  #

    return acquisition


# ====================================================================
# Variance as Constraint
class FF_MV_qECI(SingleModelAcquisitionBuilder):
    def __init__(
        self,
        noise_type,
        noise_param: TensorType,
        variance_threshold: float,
        pseudo_min: float = 1000,
        ff_method: str = "RFF",
        opt_ff_num: int = 1000,
        infer_mc_num: int = 10000,
        temperature_parameter: float = 1e-3,
        mc_num: int = 32,
        max_batch_element: int = 500,
        rec_mc_num: int = 128,
        rec_var_prob_threshold: float = 0.5,
        implicit_sample: bool = False,
    ):
        """
        :param noise_type
        :param noise_param: representing variance in case noise_type is normal
        :param ff_method
        :param opt_ff_num
        :param mc_num
        """
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._ff_method = ff_method
        self._opt_ff_num = opt_ff_num
        self._infer_mc_num = infer_mc_num
        self._mc_num = mc_num
        self._noise_type = noise_type
        self._noise_param = noise_param
        self._pseudo_min = pseudo_min
        self._variance_threshold = variance_threshold
        self._default_temperature_parameter = temperature_parameter
        self._max_batch_element = max_batch_element
        self.fmean_mean = None
        self.fvar_mean = None
        self.sampler = None
        self._rec_mc_num = rec_mc_num
        self._rec_var_prob_threshold = rec_var_prob_threshold
        self._implicit_sample = implicit_sample

    def using(self, tag: str) -> AcquisitionFunctionBuilder[T]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(AcquisitionFunctionBuilder[T]):
            def prepare_acquisition_function(
                self,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag], **kwargs
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.update_acquisition_function(
                    function,
                    models[tag],
                    dataset=None if datasets is None else datasets[tag],
                    **kwargs,
                )

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

            # Hard code: but i don't have any better solution here:
            @property
            def fmean_mean(self):
                return single_builder.fmean_mean

            @property
            def fvar_mean(self):
                return single_builder.fvar_mean

            @property
            def get_recommend(self):
                # for CO_MVA_BO, MO_MVA_BO
                return single_builder.recommend

        return _Anon()

    def recommend(self, model: ProbabilisticModel, search_candidates: TensorType = None):
        fmean_pred = self.fmean_mean(search_candidates)
        fvar_pred = tf.transpose(
            self.sampler.get_var_trajectory(
                sample_size=self._rec_mc_num, max_batch_element_num=self._max_batch_element
            )(search_candidates),
            perm=[1, 0, 2],
        )  # [mc_samples, N, 1] -> [N, mc_samples, 1]
        fvar_pred_prob = tf.reduce_mean(
            tf.cast(fvar_pred <= self._variance_threshold, dtype=tf.float64), -2
        )  # [N, 1]
        feasible_mask = tf.squeeze(fvar_pred_prob >= self._rec_var_prob_threshold)  # [N, 1]
        if not tf.reduce_any(feasible_mask):  # no feasible point at all
            recommend_x = search_candidates[tf.squeeze(tf.argmax(fvar_pred_prob))]
        else:
            recommend_x = search_candidates[feasible_mask][
                tf.squeeze(tf.argmin(fmean_pred[feasible_mask]))
            ]
        recommend_x = recommend_x[tf.newaxis] if tf.rank(recommend_x) == 1 else recommend_x
        return recommend_x

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        def infer_fmin_based_on_GP_model():
            fmean_pred = self.fmean_mean(dataset.query_points)
            fvar_pred = self.fvar_mean(dataset.query_points)
            feasible_mask = fvar_pred <= self._variance_threshold
            if not tf.reduce_any(feasible_mask):  # no feasible point at all
                inferred_min = tf.constant([self._pseudo_min], dtype=dataset.query_points.dtype)
            else:
                inferred_min = tf.reduce_min(fmean_pred[feasible_mask])
            return inferred_min

        def get_feasible_fmin_samples_from_traj(sampled_trajectories: list[Callable, Callable]):
            mean_traj, var_traj = sampled_trajectories
            mean_trajs = mean_traj(dataset.query_points)  # [mc, N, 1]
            var_trajs = var_traj(dataset.query_points)  # [mc, N, 1]
            feasible_mask = var_trajs <= self._variance_threshold  # [mc, N, 1]
            psudo_obj = tf.cast(
                feasible_mask, dataset.query_points.dtype
            ) * mean_trajs + self._pseudo_min * tf.ones_like(mean_trajs) * tf.cast(
                ~feasible_mask, dataset.query_points.dtype
            )
            return tf.reduce_min(psudo_obj, axis=-2)

        self.fmean_mean = partial(
            fmean_mean,
            model=model,
            noisy_disy=self._noise_dist,
            sample_number=self._infer_mc_num,
        )
        self.fvar_mean = partial(
            fvar_mean,
            model=model,
            noisy_disy=self._noise_dist,
            sample_number=self._infer_mc_num,
        )

        if self._ff_method == "RFF":
            self.sampler = RFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )
        else:
            assert self._ff_method == "QFF"
            self.sampler = QFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )

        # construct sampled trajectory
        fmean, theta_sample = self.sampler.get_mean_trajectory(
            sample_size=self._mc_num,
            return_sample=True,
            max_batch_element_num=self._max_batch_element,
        )
        var = self.sampler.get_var_trajectory(
            theta_sample=theta_sample, max_batch_element_num=self._max_batch_element
        )
        trajectories = [fmean, var]

        if self._implicit_sample is False:  # use GP posterior mean to inferr the mean and variance
            inferred_min = infer_fmin_based_on_GP_model()  # zero rank tensor
        else:
            # use sampled mmin
            inferred_min = get_feasible_fmin_samples_from_traj(trajectories)  # [mc, 1]

        return batch_mv_qcei(
            trajectories,
            feasible_min_mean=inferred_min,
            variance_threshold=self._variance_threshold,
            temperature_parameter=self._default_temperature_parameter,
        )


def batch_mv_qcei(
    sampled_trajectories: list,
    feasible_min_mean: TensorType,
    variance_threshold: float,
    temperature_parameter: float = 1e-9,
    pending_points: Optional[TensorType] = None,
) -> AcquisitionFunction:
    """
    :param sampled_trajectories
    :param feasible_min_mean
    :param pending_points
    :param variance_threshold
    :param temperature_parameter
    """

    def acquisition(at: TensorType) -> TensorType:
        # _batch_size = 1 if pending_points is None else pending_points.shape[0] + 1  # B
        _batch_size = at.shape[-2]

        if pending_points is not None:
            raise NotImplementedError("Greedy Batch not supported yet")
        else:  # pending point is none, we squeeze at: assume it can only be sequential here
            mean_traj, var_traj = sampled_trajectories
            # at [N ,B, dim] -> [N * B, dim] # implicitly assume at with rank 3
            _hack_at = tf.reshape(at, (tf.shape(at)[0] * tf.shape(at)[1], tf.shape(at)[-1]))
            res = tf.concat([mean_traj(_hack_at), var_traj(_hack_at)], -1)  # [_mc_num, N * B, 2]
            res = tf.reshape(
                res, (tf.shape(res)[0], tf.shape(at)[0], tf.shape(at)[1], tf.shape(res)[-1])
            )  # [_mc_num, N, B, 2]
            samples = tf.transpose(res, perm=[1, 0, 2, 3])  # [N, _mc_num, B, 2]

        sample_means, sample_vars = tf.split(samples, 2, axis=-1)  # [N, mc, B, 1], [N, mc, B, 1]

        relax_feasible_indicator = tf.sigmoid(
            (variance_threshold - sample_vars) / temperature_parameter
        )  # [N, mc, B, 1]
        un_constraint_improvement = tf.maximum(
            feasible_min_mean - sample_means, tf.zeros(shape=1, dtype=at.dtype)
        )
        # [N, mc, B, 1] -> [N, mc, 1] -> [N, 1]
        c_imprv = tf.reduce_mean(
            tf.reduce_max(relax_feasible_indicator * un_constraint_improvement, -2),
            -2,
        )
        return c_imprv

    return acquisition


class FF_SMV_qEI(SingleModelAcquisitionBuilder):
    def __init__(
        self,
        noise_type,
        noise_param: TensorType,
        pseudo_min: float = 1000,
        ff_method: str = "RFF",
        opt_ff_num: int = 1000,
        infer_mc_num: int = 10000,
        temperature_parameter: float = 1e-3,
        mc_num: int = 32,
        max_batch_element: int = 500,
        alpha_var=1.0,
        implicit_sample: bool = False,
        rec_mc_num: int = 128,
    ):
        """
        :param noise_type
        :param noise_param: representing variance in case noise_type is normal
        :param ff_method
        :param opt_ff_num
        :param mc_num
        """
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._ff_method = ff_method
        self._opt_ff_num = opt_ff_num
        self._infer_mc_num = infer_mc_num
        self._mc_num = mc_num
        self._noise_type = noise_type
        self._noise_param = noise_param
        self._pseudo_min = pseudo_min
        self._default_temperature_parameter = temperature_parameter
        self._max_batch_element = max_batch_element
        self._alpha_var = alpha_var
        self._fmean_mean = None
        self._fvar_mean = None
        self._rec_mc_num = rec_mc_num
        self._implicit_sample = implicit_sample

    def using(self, tag: str) -> AcquisitionFunctionBuilder[T]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(AcquisitionFunctionBuilder[T]):
            def prepare_acquisition_function(
                self,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag], **kwargs
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.update_acquisition_function(
                    function,
                    models[tag],
                    dataset=None if datasets is None else datasets[tag],
                    **kwargs,
                )

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

            # Hard code: but i don't have any better solution here:
            @property
            def fmean_mean(self):
                return single_builder.fmean_mean

            @property
            def fvar_mean(self):
                return single_builder.fvar_mean

            @property
            def get_recommend(self):
                # for CO_MVA_BO, MO_MVA_BO
                return single_builder.recommend

        return _Anon()

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        def get_fmin_samples_from_traj(sampled_trajectories: list[Callable, Callable]):
            mean_traj, var_traj = sampled_trajectories
            mean_trajs = mean_traj(dataset.query_points)  # [mc, N, 1]
            var_trajs = var_traj(dataset.query_points)  # [mc, N, 1]
            scalarized_trajs = mean_trajs + self._alpha_var * var_trajs  # [mc, N, 1]
            return tf.reduce_min(scalarized_trajs, axis=-2)  # [mc, 1]

        # This two are only used for posterior inference
        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )

        if self._ff_method == "RFF":
            self.sampler = RFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )
        else:
            assert self._ff_method == "QFF"
            self.sampler = QFFVarianceSampler(
                self._noise_type,
                model,
                dataset,
                num_features=self._opt_ff_num,
                noise=self._noise_param,
            )

        # construct sampled trajectory
        fmean, theta_sample = self.sampler.get_mean_trajectory(
            sample_size=self._mc_num,
            return_sample=True,
            max_batch_element_num=self._max_batch_element,
        )
        var = self.sampler.get_var_trajectory(
            theta_sample=theta_sample, max_batch_element_num=self._max_batch_element
        )
        trajectories = [fmean, var]

        # costruct min samples
        if self._implicit_sample is not True:
            # use point estimate min, var
            fmean_pred = self.fmean_mean(dataset.query_points)
            fvar_pred = self.fvar_mean(dataset.query_points)
            inferred_min = tf.reduce_min(fmean_pred + self._alpha_var * fvar_pred, axis=0)  # [1, 1]
        else:
            # use sampled mmin
            inferred_min = get_fmin_samples_from_traj(trajectories)  # [mc, 1]

        return batch_scalarization_qei(
            trajectories, best_scalarized_obj=inferred_min, alpha_var=self._alpha_var
        )

    def recommend(self, model: ProbabilisticModel, search_candidates: TensorType = None):
        """
        Calculate Probability of Better
        """
        # get mean pred min
        fmean_pred = self.fmean_mean(search_candidates)
        fvar_pred = self.fvar_mean(search_candidates)
        _point_est_obj = fmean_pred + self._alpha_var * fvar_pred
        point_estimate_min_index = tf.squeeze(tf.argmin(_point_est_obj))
        point_estimation_min_mask = tf.squeeze((_point_est_obj == tf.reduce_min(_point_est_obj)))
        #
        fmean_traj, theta_sample = self.sampler.get_mean_trajectory(
            sample_size=self._rec_mc_num,
            max_batch_element_num=self._max_batch_element,
            return_sample=True,
        )
        fmean_samples = tf.transpose(fmean_traj(search_candidates), perm=[1, 0, 2])
        fvar_samples = tf.transpose(
            self.sampler.get_var_trajectory(
                max_batch_element_num=self._max_batch_element, theta_sample=theta_sample
            )(search_candidates),
            perm=[1, 0, 2],
        )  # [mc_samples, N, 1] -> [N, mc_samples, 1]
        obj_samples = fmean_samples + self._alpha_var * fvar_samples  # [N, mc_samples, 1]
        sorted_obj_pred = tf.sort(obj_samples, axis=-2)  # [N, mc_samples, 1]
        prob_smaller = tf.reduce_mean(
            tf.cast(
                sorted_obj_pred <= sorted_obj_pred[point_estimate_min_index],
                dtype=search_candidates.dtype,
            ),
            -2,
        )
        if tf.reduce_any(
            prob_smaller[~point_estimation_min_mask] > 0.5
        ):  # no feasible point at all
            print("Activate Probability Smaller Recommendation")
            recommend_x = search_candidates[~point_estimation_min_mask][
                tf.squeeze(tf.argmax(prob_smaller[~point_estimation_min_mask]))
            ]
        else:
            recommend_x = search_candidates[point_estimate_min_index]
        recommend_x = recommend_x[tf.newaxis] if tf.rank(recommend_x) == 1 else recommend_x
        return recommend_x


def batch_scalarization_qei(
    sampled_trajectories: list,
    best_scalarized_obj: TensorType,
    alpha_var: float,
    pending_points: Optional[TensorType] = None,
):
    """
    :param sampled_trajectories
    :param best_scalarized_obj: shape [mc, 1]
    :param pending_points
    """

    def acquisition(at: TensorType) -> TensorType:
        # _batch_size = 1 if pending_points is None else pending_points.shape[0] + 1  # B
        _batch_size = at.shape[-2]

        if pending_points is not None:
            raise NotImplementedError("Greedy Batch not supported yet")
        else:  # pending point is none, we squeeze at: assume it can only be sequential here
            mean_traj, var_traj = sampled_trajectories
            # at [N ,B, dim] -> [N * B, dim] # implicitly assume at with rank 3
            _hack_at = tf.reshape(at, (tf.shape(at)[0] * tf.shape(at)[1], tf.shape(at)[-1]))
            res = tf.concat([mean_traj(_hack_at), var_traj(_hack_at)], -1)  # [_mc_num, N * B, 2]
            res = tf.reshape(
                res, (tf.shape(res)[0], tf.shape(at)[0], tf.shape(at)[1], tf.shape(res)[-1])
            )  # [_mc_num, N, B, 2]
            samples = tf.transpose(res, perm=[1, 0, 2, 3])  # [N, _mc_num, B, 2]

        sample_means, sample_vars = tf.split(samples, 2, axis=-1)  # [N, mc, B, 1], [N, mc, B, 1]

        scalarization_samples = sample_means + alpha_var * sample_vars  # [N, mc, B, 1]
        _best_scalarized_obj = tf.expand_dims(best_scalarized_obj, axis=-2)  # [mc, B, 1]
        # [N, mc, B, 1] -> [N, mc, 1] -> [N, 1]
        return tf.reduce_mean(
            tf.reduce_max(tf.maximum(_best_scalarized_obj - scalarization_samples, 0), axis=-2),
            axis=1,
        )

    return acquisition


# ==========================Implementation of iwazaki2021mean function ================
# from matplotlib import pyplot as plt
#
# plt.fill_between(tf.squeeze(search_candidates), tf.squeeze(
#     calculate_F1_lt(search_candidates, base_samples, model, self._beta_t)),
#                  tf.squeeze(calculate_F1_ut(search_candidates, base_samples, model,
#                                             self._beta_t)), label='Mean Bounds')
# plt.fill_between(tf.squeeze(search_candidates), tf.squeeze(
#     calculate_F2_lt(search_candidates, base_samples, model, self._beta_t)),
#                  tf.squeeze(calculate_F2_ut(search_candidates, base_samples, model,
#                                             self._beta_t)), label='Var Bounds')
# plt.legend()
# plt.show()


def calculate_ut(at: TensorType, model, beta_t) -> Tuple[TensorType, TensorType]:
    mu, var = model.predict(at)
    return mu + tf.cast(tf.sqrt(beta_t), dtype=at.dtype) * tf.sqrt(var)


def calculate_lt(at: TensorType, model, beta_t) -> TensorType:
    """
    :param at
    """
    mu, var = model.predict(at)
    return mu - tf.cast(tf.sqrt(beta_t), dtype=at.dtype) * tf.sqrt(var)


def calculate_F1_lt(at: TensorType, noise_base_samples, model, beta_t) -> TensorType:
    input_dists = tf.expand_dims(at, -2) + tf.cast(
        noise_base_samples, dtype=at.dtype
    )  # [N, mc, dim]
    return tf.reduce_mean(calculate_lt(input_dists, model, beta_t), -2)  # [N, 1]


def calculate_F1_ut(at: TensorType, noise_base_samples, model, beta_t) -> TensorType:
    input_dists = tf.expand_dims(at, -2) + tf.cast(
        noise_base_samples, dtype=at.dtype
    )  # [N, mc, dim]
    return tf.reduce_mean(calculate_ut(input_dists, model, beta_t), -2)  # [N, 1]


def calculate_lt_tilde(at: TensorType, model, beta_t) -> TensorType:
    """
    at: {N, noise_dist, dim}
    """
    return calculate_lt(at, model, beta_t) - tf.reduce_mean(
        calculate_ut(at, model, beta_t), -2, keepdims=True
    )


def calculate_ut_tilde(at: TensorType, model, beta_t) -> TensorType:
    """
    at: {N, noise_dist, dim}
    """
    return calculate_ut(at, model, beta_t) - tf.reduce_mean(
        calculate_lt(at, model, beta_t), -2, keepdims=True
    )


def calculate_lt_tilde_sq(at: TensorType, model, beta_t) -> TensorType:
    _lt_tilde = calculate_lt_tilde(at, model, beta_t)
    _ut_tilde = calculate_ut_tilde(at, model, beta_t)
    lt_tilde_ut_tile_between_zero_mask = tf.logical_and(
        _lt_tilde <= tf.zeros(1, dtype=at.dtype), tf.zeros(1, dtype=at.dtype) <= _ut_tilde
    )
    _min = tf.minimum(_lt_tilde ** 2, _ut_tilde ** 2)
    return _min * tf.cast(~lt_tilde_ut_tile_between_zero_mask, dtype=at.dtype)


def calculate_ut_tilde_sq(at: TensorType, model, beta_t) -> TensorType:
    """
    at: {N, noise_dist, dim}
    """
    _lt_tilde = calculate_lt_tilde(at, model, beta_t)
    _ut_tilde = calculate_ut_tilde(at, model, beta_t)
    return tf.maximum(_lt_tilde ** 2, _ut_tilde ** 2)


@sequential_batch(max_batch_element_num=50)
def calculate_F2_lt(at: TensorType, noise_base_samples, model, beta_t) -> TensorType:
    """
    calculate Standard deviation's lower bound
    """
    if tf.rank(at) == 1:
        at = tf.expand_dims(at, -1)
    input_dists = tf.expand_dims(at, -2) + tf.cast(
        noise_base_samples, dtype=at.dtype
    )  # [N, mc, dim]
    return tf.reduce_mean(calculate_lt_tilde_sq(input_dists, model, beta_t), -2)  # [N, 1]


@sequential_batch(max_batch_element_num=50)
def calculate_F2_ut(at: TensorType, noise_base_samples, model, beta_t) -> TensorType:
    """
    calculate Standard deviation's upper bound
    """
    if tf.rank(at) == 1:
        at = tf.expand_dims(at, -1)
    input_dists = tf.expand_dims(at, -2) + tf.cast(
        noise_base_samples, dtype=at.dtype
    )  # [N, mc, dim]
    return tf.reduce_mean(calculate_ut_tilde_sq(input_dists, model, beta_t), -2)  # [N, 1]


def lambda_t(
    model: ProbabilisticModel, beta_t: float, base_sample, max_batch=5
) -> AcquisitionFunction:
    """
    Eq 4 of the paper
    :param model
    :param beta_t
    """

    def acquisition(at: TensorType) -> TensorType:
        at = tf.squeeze(at, -2)  # [N, dim]

        F1_lt = calculate_F1_lt(at, base_sample, model, beta_t)  # [N, 1]
        F1_ut = calculate_F1_ut(at, base_sample, model, beta_t)  # [N, 1]
        F2_lt = calculate_F2_lt(at, base_sample, model, beta_t)  # [N, 1]
        F2_ut = calculate_F2_ut(at, base_sample, model, beta_t)  # [N, 1]

        return (F1_ut - F1_lt) ** 2 + (F2_ut - F2_lt) ** 2

    return acquisition


class MO_MVA_BO(SingleModelAcquisitionBuilder):
    """
    main reference: iwazaki2021mean

    We modified the objective a little bit, so that the 2nd objective is not -\sqrt{V} but directly V

    Here, we modify its own acquisition function (i.e., Mt and Eq. 4) to make it suitable
    for continuous problem: we use SLSQP to handle its constraint
    """

    def __init__(
        self,
        noise_type: str,
        noise_param,
        tau: float,
        beta_t: float = 2.0,
        approx_mc_num: int = 1000,
        gp_infer_mc_num: int = 10000,
        max_batch_element_num: int = 500,
    ):
        _supported_noise_type = ["normal", "uniform"]
        assert noise_type in _supported_noise_type
        noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
        if noise_type == "normal":
            self._noise_dist = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(shape=noise_param.shape[0], dtype=tf.float64),
                covariance_matrix=tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        else:
            # TODO: Test of uniform
            self._noise_dist = tfd.Uniform(
                -tf.convert_to_tensor(noise_param, dtype=tf.float64),
                tf.convert_to_tensor(noise_param, dtype=tf.float64),
            )
        self._beta_t = beta_t
        self._approx_mc_num = approx_mc_num
        self._infer_mc_num = gp_infer_mc_num
        self._max_batch_element_num = max_batch_element_num

        self._tau = tau
        self.fmean_mean = None
        self.fvar_mean = None

    @property
    def tau(self):
        return self._tau

    def using(self, tag: str) -> AcquisitionFunctionBuilder[T]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(AcquisitionFunctionBuilder[T]):
            def prepare_acquisition_function(
                self,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag], **kwargs
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[str, T],
                datasets: Optional[Mapping[str, Dataset]] = None,
                **kwargs,
            ) -> AcquisitionFunction:
                return single_builder.update_acquisition_function(
                    function,
                    models[tag],
                    dataset=None if datasets is None else datasets[tag],
                    **kwargs,
                )

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

            # Hard code: but i don't have any better solution here:
            @property
            def fmean_mean(self):
                return single_builder.fmean_mean

            @property
            def fvar_mean(self):
                return single_builder.fvar_mean

            @property
            def tau(self):
                return single_builder._tau

            @property
            def get_recommend(self):
                # for CO_MVA_BO, MO_MVA_BO
                return single_builder.recommend

        return _Anon()

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        search_candidates: TensorType = None,
    ) -> Tuple[AcquisitionFunction, DiscreteSearchSpace]:
        """
        Prepare the estimate Pareto set
        and optimize within the potential Pareto set
        """

        assert search_candidates is not None

        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        base_samples = self._noise_dist.sample(self._approx_mc_num)

        def calc_estimate_Pareto_set_mask(at: TensorType) -> bool:
            """
            Prepare pi_hat_t, the estimate Pareto set
            Which has been constructed
            """
            ut_F1 = calculate_F1_ut(at, base_samples, model, self._beta_t)
            ut_F2 = calculate_F2_ut(at, base_samples, model, self._beta_t)

            # get non-dominated
            non_dominated_mask = non_dominated(tf.concat([ut_F1, ut_F2], -1))[1] == 0
            return non_dominated_mask

        def calc_potential_Pareto_set_mask(at: TensorType) -> bool:
            lt_F1 = calculate_F1_lt(at, base_samples, model, self._beta_t)
            lt_F2 = calculate_F2_lt(at, base_samples, model, self._beta_t)

            # get non-dominated
            non_dominated_mask = non_dominated(tf.concat([lt_F1, lt_F2], -1))[-1] == 0
            return non_dominated_mask

        pi_hat_t_mask = calc_estimate_Pareto_set_mask(search_candidates)
        Mt_mask = calc_potential_Pareto_set_mask(search_candidates)

        potential_candidates = search_candidates[tf.logical_or(pi_hat_t_mask, Mt_mask)]

        return lambda_t(
            model, self._beta_t, base_samples, max_batch=self._max_batch_element_num
        ), DiscreteSearchSpace(potential_candidates)

    def update_acquisition_function(
        self, function: AcquisitionFunction, model: T, dataset: Optional[Dataset] = None, **kwargs
    ) -> AcquisitionFunction:
        return self.prepare_acquisition_function(model, dataset=dataset, **kwargs)

    def recommend(self, model: ProbabilisticModel, search_candidates: TensorType = None):

        base_samples = self._noise_dist.sample(self._approx_mc_num)

        def calc_estimate_Pareto_set_mask(at: TensorType) -> bool:
            """
            Prepare pi_hat_t, the estimate Pareto set
            Which has been constructed
            """
            ut_F1 = calculate_F1_ut(at, base_samples, model, self._beta_t)
            ut_F2 = calculate_F2_ut(at, base_samples, model, self._beta_t)

            # get non-dominated
            non_dominated_mask = non_dominated(tf.concat([ut_F1, ut_F2], -1))[1] == 0
            return non_dominated_mask

        Pi_t_mask = tf.squeeze(calc_estimate_Pareto_set_mask(search_candidates))
        recommend_x = search_candidates[Pi_t_mask]
        recommend_x = recommend_x[tf.newaxis] if tf.rank(recommend_x) == 1 else recommend_x
        return recommend_x


class CO_MVA_BO(MO_MVA_BO):
    """
    Algorithm 3 of iwazaki2021mean

    Note since we use variance other than standard deviation as the 2nd objective
    this acquisition function has been minorly modified to fulfill our problem setting
    """

    def __init__(
        self,
        noise_type,
        noise_param: TensorType,
        variance_threshold: float,
        tau: float,
        infer_mc_num: int = 10000,
        mc_num: int = 32,
        max_batch_element: int = 500,
        approx_mc_num: int = 500,
        pseudo_min: float = 1000,
        max_batch_element_num: int = 500,
        **kwargs,
    ):
        """
        :param noise_type
        :param noise_param: representing variance in case noise_type is normal
        :param ff_method
        :param opt_ff_num
        :param mc_num
        """
        self._infer_mc_num = infer_mc_num
        self._mc_num = mc_num
        self._variance_threshold = variance_threshold
        self._max_batch_element = max_batch_element
        self._pseudo_min = pseudo_min
        super().__init__(
            noise_type,
            noise_param,
            tau,
            approx_mc_num=approx_mc_num,
            max_batch_element_num=max_batch_element_num,
            **kwargs,
        )

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        search_candidates: TensorType = None,
    ) -> Tuple[AcquisitionFunction, DiscreteSearchSpace]:

        assert search_candidates is not None

        self.fmean_mean = partial(
            fmean_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        self.fvar_mean = partial(
            fvar_mean, model=model, noisy_disy=self._noise_dist, sample_number=self._infer_mc_num
        )
        base_samples = self._noise_dist.sample(self._approx_mc_num)

        def construct_St(at: TensorType) -> bool:
            return (
                calculate_F2_ut(at, base_samples, model, self._beta_t) <= self._variance_threshold
            )

        St_mask = tf.squeeze(construct_St(search_candidates))
        if tf.size(search_candidates[St_mask]) != 0:
            current_best = tf.reduce_min(
                calculate_F1_ut(search_candidates[St_mask], base_samples, model, self._beta_t)
            )
        else:
            current_best = self._pseudo_min
        Mt_cons_mask = (
            calculate_F2_lt(search_candidates, base_samples, model, self._beta_t)
            <= self._variance_threshold
        )
        Mt_obj_mask = (
            calculate_F1_lt(search_candidates, base_samples, model, self._beta_t) <= current_best
        )

        potential_candidates = search_candidates[
            tf.squeeze(tf.logical_and(Mt_cons_mask, Mt_obj_mask))
        ]

        return lambda_t(model, self._beta_t, base_samples), DiscreteSearchSpace(
            potential_candidates
        )

    def recommend(self, model: ProbabilisticModel, search_candidates: TensorType = None):

        base_samples = self._noise_dist.sample(self._approx_mc_num)

        def construct_St(at: TensorType) -> bool:
            return (
                calculate_F2_ut(at, base_samples, model, self._beta_t) <= self._variance_threshold
            )

        St_mask = tf.squeeze(construct_St(search_candidates))
        if not tf.reduce_any(St_mask):  # Nothing to recommend
            recommend_x = search_candidates[
                tf.squeeze(
                    tf.argmin(calculate_F2_ut(search_candidates, base_samples, model, self._beta_t))
                )
            ]
        else:
            F1_ut = calculate_F1_ut(search_candidates[St_mask], base_samples, model, self._beta_t)
            recommend_x = search_candidates[St_mask][tf.squeeze(tf.argmin(F1_ut))]
        recommend_x = recommend_x[tf.newaxis] if tf.rank(recommend_x) == 1 else recommend_x

        return recommend_x


class MT_MVA_BO(MO_MVA_BO):
    """
    The correctness of this acq is evaluated using:

    Given dense training data, it focus on where we think is the true optima
    # Start doe with 100 data
    from matplotlib import pyplot as plt
    search_candidates = tf.linspace([0], [1], 100)
    plt.fill_between(tf.squeeze(search_candidates), tf.squeeze(calculate_F1_lt(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t)),
                     tf.squeeze(calculate_F1_ut(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t)), label='Mean Bounds')
    plt.fill_between(tf.squeeze(search_candidates), tf.squeeze(calculate_F2_lt(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t)),
                     tf.squeeze(calculate_F2_ut(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t)), label='Var Bounds')
    plt.plot(tf.squeeze(search_candidates), tf.squeeze(
        calculate_F1_ut(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t) +
        self._alpha * calculate_F2_ut(search_candidates, self._noise_dist, self._approx_mc_num, model, self._beta_t)), label='Acq')
    plt.legend()
    plt.show()
    """

    def __init__(
        self,
        noise_type,
        noise_param: TensorType,
        alpha: float,
        infer_mc_num: int = 10000,
        mc_num: int = 32,
        max_batch_element: int = 500,
        approx_mc_num: int = 500,
        pseudo_min: float = 1000,
        max_batch_element_num: int = 500,
    ):
        """
        :param noise_type
        :param noise_param: representing variance in case noise_type is normal
        :param ff_method
        :param opt_ff_num
        :param mc_num
        """
        self._infer_mc_num = infer_mc_num
        self._mc_num = mc_num
        self._max_batch_element = max_batch_element
        self._pseudo_min = pseudo_min
        self._alpha = alpha
        super().__init__(
            noise_type,
            noise_param,
            0,
            approx_mc_num=approx_mc_num,
            max_batch_element_num=max_batch_element_num,
        )

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        search_candidates: TensorType = None,
    ) -> AcquisitionFunction:
        base_samples = self._noise_dist.sample(self._approx_mc_num)
        return neg_lt_G(model, self._beta_t, base_samples, self._alpha)

    def recommend(self, model: ProbabilisticModel, search_candidates: TensorType = None):
        base_samples = self._noise_dist.sample(self._approx_mc_num)
        F1_ut = calculate_F1_ut(search_candidates, base_samples, model, self._beta_t)  # [N, 1]
        F2_ut = calculate_F2_ut(search_candidates, base_samples, model, self._beta_t)  # [N, 1]

        # recommend_x = search_candidates[tf.squeeze(tf.argmin(self._alpha * F1_ut + (1 - self._alpha) * F2_ut))]
        # In order to compare fairly, we use the same
        recommend_x = search_candidates[tf.squeeze(tf.argmin(F1_ut + self._alpha * F2_ut))]
        recommend_x = recommend_x[tf.newaxis] if tf.rank(recommend_x) == 1 else recommend_x
        return recommend_x


def neg_lt_G(
    model: ProbabilisticModel,
    beta_t: float,
    base_samples: TensorType,
    alpha: float,
    max_batch: int = 100,
):
    """
    Since we are performing minimization: we use this form

    :param model
    :param beta_t
    """

    def acquisition(at: TensorType) -> TensorType:
        at = tf.squeeze(at, -2)  # [N, dim]

        F1_lt = calculate_F1_lt(at, base_samples, model, beta_t)  # [N, 1]
        F2_lt = calculate_F2_lt(at, base_samples, model, beta_t)  # [N, 1]

        # In order to compare fairly, we use the same
        return -(F1_lt + alpha * F2_lt)
        # return alpha * F1_lt + (1 - alpha) * F2_lt

    return acquisition
