#PADDLE TEST

Make virtual environment first

- install paddlepaddle
```
# CUDA10.2
python -m pip install paddlepaddle-gpu==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CPU
python -m pip install paddlepaddle==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

- Confirm the paddle version
python -c "import paddle; print(paddle.__version__)"

- Clone & Install paddleDetection
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Install other dependencies
cd PaddleDetection
pip install -r requirements.txt

# Compile and install paddledet
python setup.py install
```
- Test the paddle
python ppdet/modeling/tests/test_architectures.py
