import abc
import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Integer, Float
import sympy
from typing import List, Literal, Union, Any, Callable, Tuple
from .system import System
from .inclusion import interval, nat_if, jac_if, mixjac_if, i2ut, ut2i, Interval
from .control import Control, ControlledSystem
import jax_verify as jv
from functools import partial

class EmbeddingSystem (System, abc.ABC) :
    """EmbeddingSystem

    Embeds a System 
    
    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n` 
        
    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n}.

    """
    sys:System

    @abc.abstractmethod
    def E(self, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array :
        """The right hand side of the embedding system.

        Args:
            t (Union[Integer,Float]): The time of the embedding system.
            x (jax.Array): The state of the embedding system.
            *args: interval-valued control inputs, disturbance inputs, etc. Depends on parent class.

        Returns:
            jax.Array: The time evolution of the state on the upper triangle
        """

    def Ei(self, i:int, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array :
        """The right hand side of the embedding system.

        Args:
            i (int): component
            t (Union[Integer,Float]): The time of the embedding system.
            x (jax.Array): The state of the embedding system.
            *args: interval-valued control inputs, disturbance inputs, etc. Depends on parent class.

        Returns:
            jax.Array: The i-th component of the time evolution of the state on the upper triangle
        """
        return self.E(t, x, *args, **kwargs)[i]

    def f(self, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.E(t, x, *args, **kwargs)

    def fi(self, i: int, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.Ei(i, t, x, *args, **kwargs)

class InclusionEmbedding (EmbeddingSystem) :
    """EmbeddingSystem

    Embeds a System 
    
    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`,
        
    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n},

    using an Inclusion Function for the dynamics f.
    """
    sys:System
    F:Callable[..., Interval]
    Fi:List[Callable[..., Interval]]
    def __init__(self, sys:System, F:Callable[..., Interval], Fi:Callable[..., Interval] = None) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function for f.

        Args:
            sys (System): The system to be embedded
            if_transform (InclusionFunction): An inclusion function for f.
        """
        self.sys = sys
        self.F = F
        self.Fi = Fi if Fi is not None else (lambda i, t, x, *args, **kwargs : self.F(t,x,*args,**kwargs)[i])
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2 

    def E(self, t: Any, x: jax.Array, *args, **kwargs) -> jax.Array:
        if self.evolution == 'continuous' :
            n = self.sys.xlen
            ret = jnp.empty(self.xlen)
            for i in range(n) :
                _xi = jnp.copy(x).at[i+n].set(x[i])
                ret = ret.at[i].set(self.Fi(i, interval(t), ut2i(_xi), *args, **kwargs).lower)
                x_i = jnp.copy(x).at[i].set(x[i+n])
                ret = ret.at[i+n].set(self.Fi(i, interval(t), ut2i(x_i), *args, **kwargs).upper)
            return ret
        elif self.evolution == 'discrete' :
            # Convert x from ut to i, compute through F, convert back to ut.
            return i2ut(self.F(interval(t), ut2i(x), *args, **kwargs))
        else :
            raise Exception("evolution needs to be 'continuous' or 'discrete'")
        
    def Ei(self, i: int, t: Any, x: jax.Array, *args, **kwargs) -> jax.Array:
        if self.evolution == 'continuous':
            n = self.sys.xlen
            if i < n :
                _xi = jnp.copy(x).at[i+n].set(x[i])
                return self.Fi[i](interval(t), ut2i(_xi), *args, **kwargs).lower
            else :
                x_i = jnp.copy(x).at[i].set(x[i+n])
                return self.Fi[i](interval(t), ut2i(x_i), *args, **kwargs).upper
        elif self.evolution == 'discrete' :
            if i < self.sys.xlen :
                return self.Fi[i](interval(t), ut2i(x), *args, **kwargs).lower
            else :
                return self.Fi[i](interval(t), ut2i(x), *args, **kwargs).upper
        else :
            raise Exception("evolution needs to be 'continuous' or 'discrete'")


class TransformEmbedding (InclusionEmbedding) :
    def __init__(self, sys:System, if_transform = nat_if) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function transform.

        Args:
            sys (System): _description_
            if_transform (IFTransform, optional): _description_. Defaults to nat_if.
        """
        F = if_transform(sys.f)
        # def Fi (i:int, *args, **kwargs) :
        #     return F(*args, **kwargs)[i]
        super().__init__(sys, F) 
    
def if_emb (sys:System, if_transform) -> TransformEmbedding :
    """Creates an EmbeddingSystem using an inclusion function transform for the dynamics of a System.

    Args:
        sys (System): System to embed
        if_transform (IFTransform): Inclusion function transform to embed the system with.

    Returns:
        EmbeddingSystem: Embedding system from the inclusion function transform.
    """
    return TransformEmbedding(sys, if_transform)

def nat_emb (sys:System) :
    """Creates an EmbeddingSystem using the natural inclusion function of the dynamics of a System.

    Args:
        sys (System): System to embed

    Returns:
        EmbeddingSystem: Embedding system from the natural inclusion function transform.
    """
    return TransformEmbedding(sys, if_transform=nat_if)

def jac_emb (sys:System) :
    """Creates an EmbeddingSystem using the Jacobian-based inclusion function of the dynamics of a System.

    Args:
        sys (System): System to embed

    Returns:
        EmbeddingSystem: Embedding system from the Jacobian-based inclusion function transform.
    """
    return TransformEmbedding(sys, if_transform=jac_if)

def mixjac_emb (sys:System) :
    """Creates an EmbeddingSystem using the Mixed Jacobian-based inclusion function of the dynamics of a System.

    Args:
        sys (System): System to embed

    Returns:
        EmbeddingSystem: Embedding system from the Mixed Jacobian-based inclusion function transform.
    """
    return TransformEmbedding(sys, if_transform=mixjac_if)

# class InterconnectedEmbedding (EmbeddingSystem) :
#     def __init__(self, sys:System, if_transform:IFTransform = nat_if) -> None:
#         self.sys = sys
#         self.F = if_transform(sys.f)
#         self.Fi = [if_transform(partial(sys.fi, i)) for i in range(sys.xlen)]
#         self.evolution = sys.evolution
#         self.xlen = sys.xlen * 2 

