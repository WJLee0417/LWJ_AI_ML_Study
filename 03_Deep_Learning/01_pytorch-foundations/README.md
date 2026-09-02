# PyTorch Foundations

딥러닝 프로젝트에서 사용하는 텐서 연산, 자동 미분, 학습 루프, 입력·출력 shape, 이미지의 공간적 특징을 작은 예제로 확인하는 단계다.

## 학습 순서

1. [신경망 fitting](notebooks/Neural_Network_Fitting.ipynb): 순전파, MSE, 역전파, 수동 SGD
2. [다중 입력](notebooks/Multi_Input_NN.ipynb): 행렬 연산과 은닉 표현
3. [다중 출력](notebooks/Multi_Output_NN.ipynb): 여러 예측값과 출력층 shape
4. [FCN과 CNN 비교](notebooks/FCN_CNN_Compare.ipynb): 공간 구조와 합성곱의 필요성
5. [짧은 구현 과제](exercises/README.md): autograd, optimizer, overfitting, DataLoader

## 다음 단계

개념 확인이 끝나면 [MNIST FCN/CNN 벤치마크](../02_mnist-cnn-benchmark/README.md)로 이동한다. 이 폴더의 수동 학습 코드는 이후 `nn.Module`, `DataLoader`, `torch.optim`, validation, checkpoint, 혼동행렬을 갖춘 재현 가능한 실험으로 확장된다.
