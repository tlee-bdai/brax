# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probability distributions in JAX."""

import abc
import jax
import jax.numpy as jnp


class ParametricDistribution(abc.ABC):
  """Abstract class for parametric (action) distribution."""

  def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
    """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: size of the parameters for the distribution
      postprocessor: bijector which is applied after sampling (in practice, it's
        tanh or identity)
      event_ndims: rank of the distribution sample (i.e. action)
      reparametrizable: is the distribution reparametrizable
    """
    self._param_size = param_size
    self._postprocessor = postprocessor
    self._event_ndims = event_ndims  # rank of eventspython how to make wrapper class
    self._reparametrizable = reparametrizable
    assert event_ndims in [0, 1]

  @abc.abstractmethod
  def create_dist(self, parameters):
    """Creates distribution from parameters."""
    pass

  @property
  def param_size(self):
    return self._param_size

  @property
  def reparametrizable(self):
    return self._reparametrizable

  def postprocess(self, event):
    return self._postprocessor.forward(event)

  def inverse_postprocess(self, event):
    return self._postprocessor.inverse(event)

  def sample_no_postprocessing(self, parameters, seed):
    return self.create_dist(parameters).sample(seed=seed)

  def sample(self, parameters, seed):
    """Returns a sample from the postprocessed distribution."""
    return self.postprocess(self.sample_no_postprocessing(parameters, seed))

  def mode(self, parameters):
    """Returns the mode of the postprocessed distribution."""
    return self.postprocess(self.create_dist(parameters).mode())

  def log_prob(self, parameters, actions):
    """Compute the log probability of actions."""
    # if isinstance(parameters, tuple):
    #   parameters = jnp.array(parameters)
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)
    log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
    if self._event_ndims == 1:
      log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
    return log_probs

  def entropy(self, parameters, seed):
    """Return the entropy of the given distribution."""
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    entropy += self._postprocessor.forward_log_det_jacobian(
        dist.sample(seed=seed))
    if self._event_ndims == 1:
      entropy = jnp.sum(entropy, axis=-1)
    return entropy


class NormalDistribution:
  """Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

  def sample(self, seed):
    return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

  def mode(self):
    return self.loc

  def log_prob(self, x):
    log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
    log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
    return log_unnormalized - log_normalization

  def entropy(self):
    log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * jnp.ones_like(self.loc)

class CategoricalDistribution:
  """Categorical distribution."""

  def __init__(self, logits):
    self.logits = logits

  def sample(self, seed):
    idx = jax.random.categorical(seed, logits=self.logits)
    out = jnp.zeros_like(self.logits)
    if len(out.shape) ==1:
      out = out.at[idx].set(1.0)
    if len(out.shape) ==2:
      out = out.at[jnp.arange(out.shape[0]), idx].set(1.0)
    elif len(out.shape) ==3:
      i_idx = jnp.arange(out.shape[0])[:, None]  # (n, 1)
      j_idx = jnp.arange(out.shape[1])[None, :]  # (1, m)
      out = out.at[i_idx, j_idx, idx].set(1.0)
    return out

  def mode(self):
    idx = jnp.argmax(self.logits, axis=-1)
    out = jnp.zeros_like(self.logits)
    out = out.at[jnp.arange(out.shape[0]), idx].set(1.0)
    return out

  def log_prob(self, x):
    idx = jnp.argmax(x, axis=-1)
    return jnp.take_along_axis(self.logits, jnp.expand_dims(idx, axis=-1), axis=-1)

  def entropy(self):
    return - self.logits * jnp.exp(self.logits)

class CompCatNormalDistribution:
  def __init__(self, logits, loc, scale):
    self.categorical_dist = CategoricalDistribution(logits)
    self.normal_dist = NormalDistribution(loc, scale)
    self.dim_c = logits.shape[-1]
    self.dim_n = loc.shape[-1]
  
  def sample(self, seed):
    seed_c, seed_n = jax.random.split(seed)
    x_c = self.categorical_dist.sample(seed_c)
    x_n = self.normal_dist.sample(seed_n)
    return jnp.concatenate([x_c, x_n], axis=-1)

  def mode(self):
    x_c = self.categorical_dist.mode()
    x_n = self.normal_dist.mode()
    return jnp.concatenate([x_c, x_n], axis=-1)

  def log_prob(self, x):
    x_c = jax.lax.slice_in_dim(x, 0, self.dim_c, axis=-1) 
    x_n = jax.lax.slice_in_dim(x, self.dim_c, self.dim_c + self.dim_n, axis=-1) 
    log_prob_c = self.categorical_dist.log_prob(x_c)
    log_prob_n = self.normal_dist.log_prob(x_n)
    return jnp.concatenate([log_prob_c, log_prob_n], axis=-1) ## THIS IS PROBLEMATIC.. AS LOG_PROB_C IS PROBABILITY AND LOG_PROB_N IS PROB. DENSITY

  def entropy(self):
    ent_c = jnp.sum(self.categorical_dist.entropy(), axis=-1, keepdims=True)
    ent_n = self.normal_dist.entropy()
    return jnp.concatenate([ent_c, ent_n], axis=-1) ## DIFFERENTIAL ENTROPY + ENTROPY IS NOT WELL DEFINED I THINK..

class TanhBijector:
  """Tanh Bijector."""

  def forward(self, x):
    return jnp.tanh(x)

  def inverse(self, y):
    return jnp.arctanh(y)

  def forward_log_det_jacobian(self, x):
    return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))

class Identity:
  def forward(self, x):
    return x
  
  def inverse(self, x):
    return x
  
  def forward_log_det_jacobian(self, x):
    return 0.0
  
class CompIdTanh:
  def __init__(self, dim_c, dim_n):
    self.dim_c = dim_c
    self.dim_n = dim_n

  def forward(self, x):
    x_c = jax.lax.slice_in_dim(x, 0, self.dim_c, axis=-1) 
    x_n = jax.lax.slice_in_dim(x, self.dim_c, self.dim_c + self.dim_n, axis=-1)
    return jnp.concatenate([x_c, jnp.tanh(x_n)], axis=-1)
  
  def inverse(self, y):
    y_c = jax.lax.slice_in_dim(y, 0, self.dim_c, axis=-1) 
    y_n = jax.lax.slice_in_dim(y, self.dim_c, self.dim_c + self.dim_n, axis=-1)
    return jnp.concatenate([y_c, jnp.arctanh(y_n)], axis=-1)
  
  def forward_log_det_jacobian(self, x):
    x_c_dummy = jax.lax.slice_in_dim(x, 0, 1, axis=-1) 
    x_n = jax.lax.slice_in_dim(x, self.dim_c, self.dim_c + self.dim_n, axis=-1)
    return jnp.concatenate([x_c_dummy * 0.0,  2. * (jnp.log(2.) - x_n - jax.nn.softplus(-2. * x_n))], axis=-1)



class NormalTanhDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001, var_scale=1):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size = 2 * event_size,
        postprocessor=TanhBijector(),
        event_ndims=1,
        reparametrizable=True)
    self._min_std = min_std
    self._var_scale = var_scale

  def create_dist(self, parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return NormalDistribution(loc=loc, scale=scale)


class ParametricCategoricalDistribution(ParametricDistribution):
  """Categorical Distribution with Softmax"""

  def __init__(self, event_size):
    super().__init__(param_size=event_size, postprocessor=Identity(), event_ndims=1,reparametrizable=True )

  def create_dist(self, parameters):
    return CategoricalDistribution(jnp.log(jax.nn.softmax(parameters)) )

class CompCatNormalTanhDisribution(ParametricDistribution):

  def __init__(self, dim_c, dim_n, min_std=0.001, var_scale=1):
    super().__init__(
      param_size = dim_c + dim_n * 2,
      postprocessor=CompIdTanh(dim_c, dim_n),
      event_ndims=1,
      reparametrizable=True
    )
    self.dim_c = dim_c
    self.dim_n = dim_n
    self._min_std = min_std
    self._var_scale = var_scale
  
  def create_dist(self, parameters):
    logits = jax.lax.slice_in_dim(parameters, 0, self.dim_c, axis=-1)
    logits = jnp.log(jax.nn.softmax(logits))
    loc_scale = jax.lax.slice_in_dim(parameters, self.dim_c, self.dim_c + self.dim_n * 2, axis=-1)
    loc, scale = jnp.split(loc_scale, 2, axis=-1)
    scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return CompCatNormalDistribution(logits, loc, scale)