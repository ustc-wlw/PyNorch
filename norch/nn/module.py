from .parameter import Parameter
from collections import OrderedDict
from abc import ABC
import pickle
import json
import inspect
import warnings
import itertools
import functools

from ..tensor import Tensor

from typing import Tuple, Iterator, Set, Optional, Callable, Union, Any

_global_forward_pre_hooks = OrderedDict()
_global_forward_hooks = OrderedDict()
_global_backward_hooks = OrderedDict()

class Module(ABC):
    """
    Abstract class for modules
    """
    def __init__(self):
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._grads = OrderedDict()
        self._buffers = OrderedDict()
        self._non_persistent_buffers_set = set()
        self._backward_hooks = OrderedDict()  # Backward 完成后会被调用的 hook
        self._forward_hooks = OrderedDict()  # Forward 完成后会被调用的 hook
        self._forward_pre_hooks = OrderedDict()  # Forward 前会被调用的 hook
        self._state_dict_hooks = OrderedDict()  # 得到 state_dict 以后会被调用的 hook
        self._load_state_dict_pre_hooks = OrderedDict()  # load state_dict 前会被调用的 hook
        self.training = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def train(self):
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield self, name, value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        yield from self._modules.values()

    def gradients(self):
        for module in self.modules():
            yield module._grads

    # def requires_grad_(self, requires_grad: bool = True):
    #     for _, _, parameter in self.parameters():
    #         parameter.requires_grad_(requires_grad)

    def zero_grad(self):
        for _, _, parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device):
        for module, name, _ in self.parameters():
            parameter = getattr(module, name)
            parameter = parameter.to(device)
            setattr(module, name, parameter)

        return self
    
    def state_dict(self):
        state = OrderedDict()
        for i, param in enumerate(self.parameters()):
            state[f'param{i}'] = param.tolist()
        return state
    
    def load_state(self, state_dict):
        for i, param in self.parameters():
            data = state_dict[f'param{i}']
            if param.shape != data.shape:
                warnings.warn(f"The 'state_dict' shape does not match model's parameter shape. "
                              f"Got {data.shape}, expected {param.shape}.")
            param.data = Parameter(data=data)

    def save(self, filename='model.pickle'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_dict(self, filename='state_dict.json'):
        state = self.state_dict()
        with open(filename, 'w') as f:
            json.dump(state, f)

    def inner_repr(self):
        return ""

    def __repr__(self):
        string = f"{self.get_name()}("
        tab = "   "
        modules = self._modules
        if modules == {}:
            string += f'\n{tab}(parameters): {self.inner_repr()}'
        else:
            for key, module in modules.items():
                string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
        return f'{string}\n)'
    
    def get_name(self):
        return self.__class__.__name__
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value
            
    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)
    
    def _apply(self, fn: Callable):
        # 对子模块进行递归调用
        for module in self.children():
            module._apply(fn)

        # 处理参数及其gradint
        for key, param in self._params.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                # with torch.no_grad():
                param_applied = fn(param)
                assert isinstance(param, Parameter)
                assert param.is_leaf
                self._params[key] = Parameter(param_applied, param.requires_grad)
                if param.grad is not None:
                    # with torch.no_grad():
                    grad_applied = fn(param.grad)
                    assert param.grad.is_leaf
                    self._params[key].grad = grad_applied.requires_grad_(param.grad.requires_grad)

        # 处理 buffers
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self
    
    def apply(self, fn: Callable):
        for module in self.children():
            module.apply(fn)
        fn(self)
        
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        gen = self._named_members(
            lambda module: module._params.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = ''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
                    
    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._params.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]
        return sorted(keys)
    
    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_params' in self.__dict__:
            _params = self.__dict__['_params']
            if name in _params:
                return _params[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise Exception("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
        
    def _call_impl(self, *input, **kwargs):
        for hook in itertools.chain(
                _global_forward_pre_hooks.values(),
                self._forward_pre_hooks.values()):
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result

        result = self.forward(*input, **kwargs)

        for hook in itertools.chain(
                _global_forward_hooks.values(),
                self._forward_hooks.values()):
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result

        if (len(self._backward_hooks) > 0) or (len(_global_backward_hooks) > 0):
            var = result
            while not isinstance(var, Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in itertools.chain(
                        _global_backward_hooks.values(),
                        self._backward_hooks.values()):
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result

    __call__ : Callable[..., Any] = _call_impl
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._params.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                    'the shape in current model is {}.'
                                    .format(key, input_param.shape, param.shape))
                    continue

                try:
                    # with torch.no_grad():
                    param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                    'whose dimensions in the model are {} and '
                                    'whose dimensions in the checkpoint are {}, '
                                    'an exception occurred : {}.'
                                    .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)