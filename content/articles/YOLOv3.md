+++
title = '4. YOLOv3: Khởi Nguồn Của Kỷ Nguyên Object Detection Hiện Đại'
date = 2024-06-25
draft = false
+++

## Tab of content
- [Tab of content](#tab-of-content)
  - [1. Lời mở đầu: Bước chân vào thế giới Object Detection](#1-loi-mo-dau-buoc-chan-vao-the-gioi-object-detection)
  - [2. Sự ra đời của YOLOv3 và Bức tranh Object Detection](#2-su-ra-doi-cua-yolov3-va-buc-tranh-object-detection)
  - [3. Kiến trúc mạng Darknet-53: Sức mạnh từ Residual Connections](#3-kien-truc-mang-darknet-53-suc-manh-tu-residual-connections)
  - [4. Dự đoán đa tỉ lệ (Multi-Scale Predictions): Khắc phục bài toán vật thể nhỏ](#4-du-doan-da-ti-le-multi-scale-predictions-khac-phuc-bai-toan-vat-the-nho)
  - [5. Cơ chế Anchor Box và Dự đoán Bounding Box](#5-co-che-anchor-box-va-du-doan-bounding-box)
    - [5.1 Tìm 9 Anchor Boxes bằng K-Means](#51-tim-9-anchor-boxes-bang-k-means)
    - [5.2 Công thức chuyển đổi tọa độ Bounding Box](#52-cong-thuc-chuyen-doi-toa-do-bounding-box)
  - [6. Bản chất toán học: Hàm mất mát (Loss Function)](#6-ban-chat-toan-hoc-ham-mat-mat-loss-function)
    - [6.1 Loss tọa độ Bounding Box](#61-loss-toa-do-bounding-box)
    - [6.2 Loss độ tin cậy vật thể (Objectness Loss)](#62-loss-do-tin-cay-vat-the-objectness-loss)
    - [6.3 Loss phân loại lớp (Classification Loss)](#63-loss-phan-loai-lop-classification-loss)
    - [6.4 Tổng hợp hàm mất mát](#64-tong-hop-ham-mat-mat)
  - [7. Nhìn nhận cá nhân: Ưu nhược điểm và Giá trị học tập](#7-nhin-nhan-ca-nhan-uu-nhuoc-diem-va-gia-tri-hoc-tap)

---

<a id="1-loi-mo-dau-buoc-chan-vao-the-gioi-object-detection"></a>
### 1. Lời mở đầu: Bước chân vào thế giới Object Detection

Chào mọi người! Sau hai bài viết trước về những thuật toán học máy cơ bản như [Linear Regression](/articles/linear_regression) và [K-Nearest Neighbors](/articles/k-nearest_neighbors), hôm nay mình muốn cùng các bạn bước sang một chương mới thú vị hơn rất nhiều: **Deep Learning trong Thị giác máy tính (Computer Vision)**.

Là một sinh viên đam mê công nghệ và yêu thích trí tuệ nhân tạo, khi bắt đầu tìm hiểu về bài toán phát hiện vật thể (Object Detection), câu hỏi đầu tiên khiến mình tò mò là: *Làm thế nào mà máy tính vừa nhận biết được đồ vật trong ảnh, vừa vẽ được khung bao chính xác theo thời gian thực (real-time)?*

Cái tên đầu tiên dẫn lối mình vào thế giới này chính là dòng họ **YOLO (You Only Look Once)**, và đặc biệt là phiên bản **YOLOv3**. 

Ở thời điểm hiện tại, cộng đồng đã phát triển thêm rất nhiều phiên bản mới như YOLOv8 hay YOLOv10 và mới nhất là YOLOv27 với vô số kỹ thuật tối ưu hóa hiện đại. Thế nhưng, với mình, YOLOv3 vẫn là mô hình mang tính biểu tượng và đáng học nhất đối với một sinh viên:
- Nó không quá trừu tượng để làm nản lòng người mới bắt đầu.
- Nó hội tụ đầy đủ những phát kiến kinh điển nhất: mạng xương sống Darknet-53, cơ chế kim tự tháp đa tỉ lệ (FPN-like), phân cụm Anchor box bằng K-Means, và cách chuyển đổi từ Softmax sang Binary Cross-Entropy để gán đa nhãn.

Bài viết này là cuốn sổ tay ghi chép lại hành trình mình tìm hiểu và bóc tách YOLOv3 — từ lý thuyết, bản chất toán học, cho đến những tư duy kiến trúc thực tế. Hy vọng những chia sẻ này sẽ là tài liệu tham khảo hữu ích cho các bạn sinh viên cũng đang trên con đường khám phá AI như mình!

---

<a id="2-su-ra-doi-cua-yolov3-va-buc-tranh-object-detection"></a>
### 2. Sự ra đời của YOLOv3 và Bức tranh Object Detection

Vào tháng 04 năm 2018, **Joseph Redmon** và **Ali Farhadi** (Đại học Washington) công bố bài báo khoa học nổi tiếng mang tên: *"YOLOv3: An Incremental Improvement"*. Tác giả đặt tiêu đề một cách rất hóm hỉnh và khiêm tốn rằng đây chỉ là một "báo cáo kỹ thuật nhỏ", nhưng thực chất nó đã tạo nên một cơn địa chấn trong cộng đồng Computer Vision.

Để hiểu được giá trị của YOLOv3, ta cần nhìn lại sự tiến hóa của dòng họ YOLO:

| Phiên bản | Năm | Đặc điểm then chốt | Điểm hạn chế lớn nhất |
| :--- | :--- | :--- | :--- |
| **YOLOv1** | 2015 | Khởi xướng Single-stage: Chia lưới $7 \times 7$, dự đoán 2 hộp mỗi ô. | Định vị kém (Localization error), cực kỳ tệ với vật thể nhỏ hoặc vật thể đứng sát nhau theo bầy đàn. |
| **YOLOv2 (YOLO9000)** | 2016 | Đưa vào **Anchor Boxes**, Batch Normalization, Darknet-19, cơ chế Passthrough layer. | Vẫn gặp khó khăn lớn khi kích thước vật thể chênh lệch quá nhiều; chỉ dự đoán ở 1 scale cuối cùng ($13 \times 13$). |
| **YOLOv3** | 2018 | **Darknet-53 (Residual Blocks)**, **Dự đoán tại 3 tỉ lệ khác nhau**, **Multi-label classification**. | Tốc độ chậm hơn YOLOv2 một chút do mạng sâu hơn, nhưng độ chính xác (mAP) tăng vọt ngang hàng với RetinaNet và Faster R-CNN. |

YOLOv3 ra đời đã giải quyết triệt để bài toán: **Làm sao để một mô hình vừa bắt được vật thể to, vừa không bỏ sót vật thể li ti, mà vẫn duy trì tốc độ trên 30 FPS trên các GPU thương mại?**

---

<a id="3-kien-truc-mang-darknet-53-suc-manh-tu-residual-connections"></a>
### 3. Kiến trúc mạng Darknet-53: Sức mạnh từ Residual Connections

Trái tim trích xuất đặc trưng của YOLOv3 là mạng **Darknet-53** (bao gồm 53 lớp tích chập Convolutional Layers). Đây là sự kết hợp tài tình giữa triết lý thiết kế của Darknet-19 (từ YOLOv2) và cơ chế kết nối phần dư **Residual Blocks** (lấy cảm hứng từ ResNet).

![Kiến trúc tổng thể YOLOv3](/images/yolov3-architecture.svg "Kiến trúc tổng thể YOLOv3 và Darknet-53")

*Hình 1: Sơ đồ toàn cảnh kiến trúc YOLOv3 — Thể hiện chi tiết kích thước Kernel (K), bước nhảy Stride (S), viền Padding (P), số bộ lọc (Filters) và dòng chảy dữ liệu đến 3 đầu ra dự đoán.*

> 💡 **Quy tắc tính kích thước Feature Map trong sơ đồ:**
> Kích thước ma trận đầu ra được tính theo công thức:
> $$ O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 $$
> - **Giữ nguyên kích thước ($S = 1, P = 1, K = 3$):** $O = \frac{416 - 3 + 2}{1} + 1 = 416$.
> - **Giảm một nửa / Downsampling ($S = 2, P = 1, K = 3$):** $O = \lfloor \frac{416 - 3 + 2}{2} \rfloor + 1 = 208$.
> - **Số kênh (Channels):** Luôn bằng đúng **số lượng bộ lọc (Filters / Number of Kernels)** của lớp đó (ví dụ: 32, 64, 128, 256, 512, 1024).

#### Bảng thông số chi tiết kiến trúc Darknet-53 (Table 1 từ Paper gốc):

| Lớp (Type) | Số bộ lọc (Filters) | Kích thước (Size) | Kích thước đầu ra (Output) | Số lần lặp (Repeat) |
| :--- | :---: | :---: | :---: | :---: |
| **Convolutional** | 32 | $3 \times 3$ | $416 \times 416$ | - |
| **Convolutional** | 64 | $3 \times 3$ / stride 2 | $208 \times 208$ | - |
| Convolutional | 32 | $1 \times 1$ | | |
| Convolutional | 64 | $3 \times 3$ | | |
| **Residual Block** | - | - | $208 \times 208$ | **$1\times$** |
| **Convolutional** | 128 | $3 \times 3$ / stride 2 | $104 \times 104$ | - |
| Convolutional | 64 | $1 \times 1$ | | |
| Convolutional | 128 | $3 \times 3$ | | |
| **Residual Block** | - | - | $104 \times 104$ | **$2\times$** |
| **Convolutional** | 256 | $3 \times 3$ / stride 2 | $52 \times 52$ | - |
| Convolutional | 128 | $1 \times 1$ | | |
| Convolutional | 256 | $3 \times 3$ | | |
| **Residual Block** | - | - | $52 \times 52$ | **$8\times$** *(Nối nhánh Scale 3)* |
| **Convolutional** | 512 | $3 \times 3$ / stride 2 | $26 \times 26$ | - |
| Convolutional | 256 | $1 \times 1$ | | |
| Convolutional | 512 | $3 \times 3$ | | |
| **Residual Block** | - | - | $26 \times 26$ | **$8\times$** *(Nối nhánh Scale 2)* |
| **Convolutional** | 1024 | $3 \times 3$ / stride 2 | $13 \times 13$ | - |
| Convolutional | 512 | $1 \times 1$ | | |
| Convolutional | 1024 | $3 \times 3$ | | |
| **Residual Block** | - | - | $13 \times 13$ | **$4\times$** *(Nối nhánh Scale 1)* |

#### Các đặc điểm kiến trúc đáng chú ý của Darknet-53:
1. **Không sử dụng Pooling Layers:** Darknet-53 loại bỏ hoàn toàn các lớp Max-Pooling. Việc giảm kích thước không gian (Downsampling) được thực hiện hoàn toàn bởi các lớp **Convolution với Stride = 2**. Điều này giúp mạng tự học cách nén thông tin không gian thay vì vứt bỏ pixel một cách cơ học.
2. **Residual Connections (Skip Connections):** Mỗi Residual block bao gồm một lớp tích chập $1 \times 1$ (để nén số kênh), theo sau bởi một lớp $3 \times 3$ (để trích xuất đặc trưng), và cộng trực tiếp đầu vào ban đầu vào đầu ra: 
   $$ y = \mathcal{F}(x, \{W_i\}) + x $$
   Cơ chế này giải quyết triệt để hiện tượng suy biến gradient (Vanishing Gradient) khi mạng lên tới hàng chục lớp.
3. **Hiệu năng vượt trội:** Trong bài báo gốc, Joseph Redmon chứng minh Darknet-53 có độ chính xác tương đương **ResNet-101** nhưng tốc độ xử lý **nhanh gấp 1.5 lần**, và tương đương **ResNet-152** nhưng **nhanh gấp 2 lần** trên cùng phần cứng.

---

<a id="4-du-doan-da-ti-le-multi-scale-predictions-khac-phuc-bai-toan-vat-the-nho"></a>
### 4. Dự đoán đa tỉ lệ (Multi-Scale Predictions): Khắc phục bài toán vật thể nhỏ

Điểm yếu chí mạng của YOLOv1 và YOLOv2 là chỉ dự đoán tại một lớp đặc trưng duy nhất ở cuối mạng ($7 \times 7$ hoặc $13 \times 13$). Khi một ảnh $416 \times 416$ bị nén xuống $13 \times 13$ (giảm $32$ lần), một vật thể nhỏ có kích thước $16 \times 16$ pixel sẽ bị co lại chưa đầy nửa pixel trên feature map, dẫn đến việc mất hoàn toàn dấu vết.

YOLOv3 mượn ý tưởng từ mạng kim tự tháp đặc trưng (**Feature Pyramid Networks - FPN**) để dự đoán độc lập tại **3 tỉ lệ khác nhau**:

![Cơ chế Kim tự tháp đa tỉ lệ trong YOLOv3](/images/yolov3-fpn-multiscale.svg "Cơ chế Feature Pyramid Network (FPN) và dự đoán đa tỉ lệ trong YOLOv3")

*Hình 2: Sơ đồ chi tiết cơ chế FPN trong YOLOv3 — Kết hợp đặc trưng sâu từ Darknet-53 với các tầng nông thông qua Upsample 2x và Concatenate để dự đoán tại 3 tỉ lệ: Layer 82 ($13\times13$), Layer 94 ($26\times26$) và Layer 106 ($52\times52$).*

1. **Scale 1 ($13 \times 13$, Stride 32):** Feature map nằm sâu nhất, trường tiếp nhận (Receptive Field) lớn nhất, chứa nhiều thông tin ngữ nghĩa trừu tượng cao $\rightarrow$ Chuyên trị các **vật thể lớn** (xe tải, xe bus, người đứng gần).
2. **Scale 2 ($26 \times 26$, Stride 16):** Lấy feature map từ Scale 1, đưa qua tích chập và phóng to kích thước (Upsampling $2\times$), sau đó ghép kênh (Concatenate) với feature map tầng tương ứng từ Darknet-53 $\rightarrow$ Chuyên trị các **vật thể kích thước trung bình** (xe hơi, bàn ghế).
3. **Scale 3 ($52 \times 52$, Stride 8):** Tiếp tục Upsampling $2\times$ và nối tiếp với tầng nông hơn của Darknet-53 $\rightarrow$ Giữ được độ phân giải không gian chi tiết $\rightarrow$ Bắt chính xác các **vật thể nhỏ** (biển báo xa, điện thoại, người đi bộ nhỏ).

#### Kích thước Tensor đầu ra (Output Tensor):
Tại mỗi ô trong lưới $S \times S$, mô hình sử dụng **3 Anchor Boxes**. Với mỗi Anchor Box, mô hình cần dự đoán:
- 4 tham số tọa độ hộp: $t_x, t_y, t_w, t_h$
- 1 điểm tin cậy vật thể (Objectness Score): $P_0$
- $C$ xác suất phân loại lớp (Class Probabilities) — ví dụ tập COCO có $C = 80$ lớp.

Do đó, số lượng kênh (Channels) tại mỗi scale là:
$$ \text{Channels} = 3 \times (4 + 1 + C) = 3 \times (4 + 1 + 80) = 255 $$

Tổng số lượng Bounding Box ứng viên mà YOLOv3 dự đoán cho một bức ảnh $416 \times 416$ là:
$$ (13 \times 13 \times 3) + (26 \times 26 \times 3) + (52 \times 52 \times 3) = 507 + 2,028 + 8,112 = 10,647 \text{ boxes} $$
*(So với chỉ 845 boxes ở YOLOv2, số lượng ứng viên tăng hơn 12 lần, giải thích vì sao YOLOv3 bắt vật thể nhỏ tốt hơn vượt bậc).*

---

<a id="5-co-che-anchor-box-va-du-doan-bounding-box"></a>
### 5. Cơ chế Anchor Box và Dự đoán Bounding Box

#### 5.1 Tìm 9 Anchor Boxes bằng K-Means
Thay vì chọn kích thước hộp mẫu (Anchor Boxes) một cách cảm tính, tác giả sử dụng thuật toán **K-Means Clustering** trên toàn bộ tập dữ liệu huấn luyện (COCO). 

Khoảng cách trong K-Means không dùng khoảng cách Euclid thông thường (vì hộp lớn sẽ bị phạt nặng hơn hộp nhỏ), mà dùng chỉ số IoU:
$$ d(\text{box}, \text{centroid}) = 1 - IoU(\text{box}, \text{centroid}) $$

Kết quả phân cụm tìm ra **9 kích thước Anchor Box chuẩn** ($width \times height$), được chia đều cho 3 scale theo thứ tự từ nhỏ đến lớn:
- **Scale 3 ($52 \times 52$ - Vật thể nhỏ):** $(10 \times 13)$, $(16 \times 30)$, $(33 \times 23)$
- **Scale 2 ($26 \times 26$ - Vật thể vừa):** $(30 \times 61)$, $(62 \times 45)$, $(59 \times 119)$
- **Scale 1 ($13 \times 13$ - Vật thể lớn):** $(116 \times 90)$, $(156 \times 198)$, $(373 \times 326)$

#### 5.2 Công thức chuyển đổi tọa độ Bounding Box
Mô hình mạng nơ-ron không trực tiếp dự đoán tọa độ pixel tuyệt đối $(x, y, w, h)$ vì giá trị dao động quá lớn sẽ khiến gradient bất ổn định. Thay vào đó, mạng dự đoán 4 giá trị độ lệch (offsets): $t_x, t_y, t_w, t_h$.

Tọa độ thực tế của Bounding Box trên ảnh được ánh xạ thông qua hệ công thức:

$$
\begin{aligned}
b_x &= \sigma(t_x) + c_x \\
b_y &= \sigma(t_y) + c_y \\
b_w &= p_w e^{t_w} \\
b_h &= p_h e^{t_h}
\end{aligned}
$$

Trong đó:
- $(c_x, c_y)$ là tọa độ góc trên bên trái của ô lưới (grid cell) hiện tại.
- $\sigma(t) = \frac{1}{1 + e^{-t}}$ là hàm Sigmoid. Nhờ hàm Sigmoid, giá trị $\sigma(t_x), \sigma(t_y) \in (0, 1)$, đảm bảo **tâm của Bounding Box luôn bị khóa chặt bên trong ô lưới đó**, không bị trôi tự do ra ngoài.
- $p_w, p_h$ là chiều rộng và chiều cao định sẵn của Anchor Box.
- Hàm mũ $e^{t_w}, e^{t_h}$ đảm bảo chiều rộng và chiều cao dự đoán luôn luôn là một số dương.

```
+---------------------+
| (cx, cy)            |
|   *-----> bx        |
|   |                 |
|   v                 |
|   by    [ BBox ]    |
|                     |
+---------------------+
```

---

<a id="6-ban-chat-toan-hoc-ham-mat-mat-loss-function"></a>
### 6. Bản chất toán học: Hàm mất mát (Loss Function)

Một cải tiến toán học then chốt trong YOLOv3 so với các phiên bản tiền nhiệm là: **Loại bỏ hoàn toàn hàm Softmax cho phân loại nhãn và chuyển sang dùng Binary Cross-Entropy độc lập.**

Lý do: Trong thực tế, các lớp không phải lúc nào cũng loại trừ lẫn nhau (mutually exclusive). Một vật thể có thể vừa mang nhãn `Woman` (Phụ nữ) vừa mang nhãn `Person` (Người). Hàm Softmax ép tổng xác suất bằng 1 nên không giải quyết được bài toán gán đa nhãn (Multi-label Classification).

Hàm mất mát tổng thể $\mathcal{L}_{total}$ của YOLOv3 là tổng có trọng số của 3 thành phần:

$$
\mathcal{L}_{total} = \mathcal{L}_{coord} + \mathcal{L}_{obj} + \mathcal{L}_{cls}
$$

#### 6.1 Loss tọa độ Bounding Box ($\mathcal{L}_{coord}$)
Sử dụng tổng bình phương sai số (Sum of Squared Errors - SSE):

$$
\mathcal{L}_{coord} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{obj} \left[ (\sigma(t_x) - \hat{x})^2 + (\sigma(t_y) - \hat{y})^2 + (t_w - \hat{w})^2 + (t_h - \hat{h})^2 \right]
$$

- $\mathbb{I}_{ij}^{obj} = 1$ nếu Anchor Box thứ $j$ trong ô lưới $i$ chịu trách nhiệm dự đoán vật thể ground-truth (có IoU lớn nhất với ground-truth), ngược lại bằng $0$.
- $\hat{x}, \hat{y}, \hat{w}, \hat{h}$ là tọa độ ground-truth đã được chuẩn hóa ngược về không gian của mạng.

#### 6.2 Loss độ tin cậy vật thể (Objectness Loss - $\mathcal{L}_{obj}$)
Được tính bằng Binary Cross-Entropy (BCE) cho cả các ô chứa vật thể và các ô nền trống:

$$
\begin{aligned}
\mathcal{L}_{obj} = &- \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{obj} \left[ \log(C_{ij}) \right] \\
&- \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{noobj} \left[ \log(1 - C_{ij}) \right]
\end{aligned}
$$

- $C_{ij}$ là giá trị dự đoán của mạng qua hàm Sigmoid $\sigma(t_o)$.
- Với các hộp không chứa vật thể ($\mathbb{I}_{ij}^{noobj}$), hệ số phạt $\lambda_{noobj}$ thường được đặt nhỏ (ví dụ $0.5$) để tránh việc các ô nền trống áp đảo hoàn toàn tín hiệu học của các ô chứa vật thể (Class Imbalance).

#### 6.3 Loss phân loại lớp (Classification Loss - $\mathcal{L}_{cls}$)
Sử dụng Binary Cross-Entropy Loss cho từng lớp riêng biệt:

$$
\mathcal{L}_{cls} = - \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{obj} \sum_{c \in \text{classes}} \left[ \hat{p}_c \log(p_c) + (1 - \hat{p}_c) \log(1 - p_c) \right]
$$

Trong đó $p_c = \sigma(t_c)$ là xác suất thuộc lớp $c$ sau khi đi qua hàm Sigmoid.

---

<a id="7-nhin-nhan-ca-nhan-uu-nhuoc-diem-va-gia-tri-hoc-tap"></a>
### 7. Nhìn nhận cá nhân: Ưu nhược điểm và Giá trị học tập

Đứng ở góc nhìn của một sinh viên đang học tập và nghiên cứu AI, việc phân tích khách quan cả điểm mạnh lẫn điểm yếu giúp ta hiểu rõ vì sao công nghệ lại liên tục phát triển.

#### 1. So sánh tương quan (Trade-offs)
- **So với Faster R-CNN (Two-stage):** Faster R-CNN có thể nhỉnh hơn 1-2% mAP ở các bài toán đặc thù yêu cầu ranh giới bounding box cực kỳ khắt khe, nhưng tốc độ chỉ đạt khoảng 5-7 FPS. YOLOv3 đạt 30-45 FPS trên GPU, hoàn toàn đủ điều kiện chạy Real-time trên luồng camera giám sát thông thường.
- **So với SSD (Single-shot MultiBox Detector):** YOLOv3 vượt trội hơn SSD ở độ chính xác trên vật thể nhỏ nhờ kiến trúc FPN kết nối nhiều tầng sâu, trong khi SSD thời kỳ đầu chỉ lấy các feature map tầng nông trực tiếp mà không có đường truyền ngược kết hợp ngữ nghĩa.

#### 2. Giới hạn kỹ thuật của YOLOv3
- **Tính toán cồng kềnh hơn YOLOv2:** Darknet-53 có số lượng tham số tương đối lớn (~62 triệu parameters), khiến việc huấn luyện từ đầu (train from scratch) trên các máy tính cá nhân của sinh viên tốn khá nhiều thời gian và tài nguyên GPU.
- **Anchor Box phụ thuộc K-Means:** Kích thước anchor được cố định theo tập dữ liệu mẫu (COCO). Khi áp dụng vào các bài toán có hình dạng vật thể dị biệt (như ảnh viễn thám hay chi tiết máy móc dài hẹp), ta bắt buộc phải chạy lại thuật toán K-Means để tìm anchor mới. *(Đây chính là lý do mà các thế hệ sau này như YOLOv8 hay YOLOv9 đã chuyển hẳn sang cơ chế Anchor-Free).*

> **Lời kết:**
> Với mình, việc học YOLOv3 không chỉ đơn thuần là biết cách gọi một thư viện nhận diện vật thể, mà là cơ hội học hỏi tư duy thiết kế hệ thống của các nhà nghiên cứu đi trước: Làm thế nào để kết hợp hài hòa giữa các khối Residual, mạng kim tự tháp FPN và các hàm mất mát để tạo nên một mô hình vừa nhanh vừa chính xác. Hiểu thấu đáo YOLOv3 chính là bước đệm vững chắc nhất để mình tự tin tiếp cận bất kỳ kiến trúc thị giác máy tính hiện đại nào sau này!

---

→ [Quay lại danh sách bài viết](/articles)
