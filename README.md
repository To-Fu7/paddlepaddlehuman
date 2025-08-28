#PADDLE TEST

Make virtual environment first

- install paddlepaddle
```
# CUDA10.2
python -m pip install paddlepaddle-gpu==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CPU
python -m pip install paddlepaddle==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

```
or check this link for more installation for CUDA : https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html

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
```
python ppdet/modeling/tests/test_architectures.py
```

- Test infer
```
# Bash
python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml \
                      --infer_img=demo/000000014439.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \

# One Line
python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml --infer_img=demo/000000014439.jpg --output_dir=infer_output/ --draw_threshold=0.5
```
