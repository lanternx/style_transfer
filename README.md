# style_transfer
style transfer script using keras(backend with tensorflow2.0

It is rewrited from https://github.com/misgod/fast-neural-style-keras to suit the state-of-art keras API. 
But there are still some problems to be fixed. 

```python
tensorflow.python.framework.errors_impl.FailedPreconditionError:  Error while reading resource variable _AnonymousVar124 from Container: localhost. This could mean that the variable was uninitialized. Not found: Resource localhost/_AnonymousVar124/class tensorflow::Var does not exist.
         [[node loss/block5_pool_loss/dummy_loss/weighted_loss/ExpandDims/ReadVariableOp (defined at D:\Tool\python3\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_16987]

Function call stack:
keras_scratch_graph
```
