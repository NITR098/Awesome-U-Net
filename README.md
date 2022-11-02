# Awesome U-Net

Official repo for Medical Image Segmentation Review: U-Net Is All You Need

# Train and Test



## Pretrained model weights

Here you can download pre-trained weights for networks.

| Network            | SegPC 2021 | ISIC 2018 | *Description*                           |
| ------------------ | ---------- | --------- | --------------------------------------- |
| **U-Net**          |            |           | Without batch normalization; 100 Epochs |
| **Att-UNet**       |            |           |                                         |
| **U-Net++**        |            |           |                                         |
| **MultiResUNet**   |            |           |                                         |
| **Residual U-Net** |            |           |                                         |
| **TransUNet**      |            |           |                                         |
| **UCTransNet**     |            |           |                                         |
| **MISSFormer**     |            |           |                                         |

# Results

Performance comparison on ***ISIC 2018*** dataset (best results are bolded).

| Methods            | AC         | PR         | SE         | SP         | Dice       | IoU        |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **U-Net**          | 0.9446     | 0.8746     | 0.8603     | 0.9671     | 0.8674     | 0.8491     |
| **Att-UNet**       | 0.9516     | 0.9075     | 0.8579     | 0.9766     | 0.8820     | 0.8649     |
| **U-Net++**        | 0.9517     | 0.9067     | 0.8590     | 0.9764     | 0.8822     | 0.8651     |
| **MultiResUNet**   | 0.9473     | 0.8765     | 0.8689     | 0.9704     | 0.8694     | 0.8537     |
| **Residual U-Net** | 0.9468     | 0.8753     | 0.8659     | 0.9688     | 0.8689     | 0.8509     |
| **TransUNet**      | 0.9452     | 0.8823     | 0.8578     | 0.9653     | 0.8499     | 0.8365     |
| **UCTransNet**     | **0.9546** | **0.9100** | **0.8704** | **0.9770** | **0.8898** | **0.8729** |
| **MISSFormer**     | 0.9453     | 0.8964     | 0.8371     | 0.9742     | 0.8657     | 0.8484     |

Performance comparison on ***SegPC 2021*** dataset (best results are bolded).

| Methods            | AC         | PR         | SE         | SP         | Dice       | IoU        |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **U-Net**          | 0.9795     | 0.9084     | 0.8548     | 0.9916     | 0.8808     | 0.8824     |
| **Att-UNet**       | 0.9854     | 0.9360     | 0.8964     | 0.9940     | 0.9158     | 0.9144     |
| **U-Net++**        | 0.9845     | 0.9328     | 0.8887     | 0.9938     | 0.9102     | 0.9092     |
| **MultiResUNet**   | 0.9753     | 0.8391     | 0.8925     | 0.9834     | 0.8649     | 0.8676     |
| **Residual U-Net** | 0.9743     | 0.8920     | 0.8080     | 0.9905     | 0.8479     | 0.8541     |
| **TransUNet**      | 0.9702     | 0.8678     | 0.7831     | 0.9884     | 0.8233     | 0.8338     |
| **UCTransNet**     | **0.9857** | **0.9365** | **0.8991** | **0.9941** | **0.9174** | **0.9159** |
| **MISSFormer**     | 0.9663     | 0.8152     | 0.8014     | 0.9823     | 0.8082     | 0.8209     |

## Visualization

- **Results on ISIC 2018**
  
  ![isic2018.png](/Users/afshin/Desktop/isic2018.png)
  
  Visual comparisons of different methods on the *ISIC 2018* skin lesion segmentation dataset. Ground truth boundaries are shown in <span style="color: #0F0">green</span>, and predicted boundaries are shown in <span style="color:blue">blue</span>.

- **Result on SegPC 2021**
  
  ![segpc.png](/Users/afshin/Desktop/segpc.png)
  
  Visual comparisons of different methods on the *SegPC 2021* cell segmentation dataset. <span style="color:red">Red</span> region indicates the Cytoplasm and <span style="color:blue">blue</span> denotes the Nucleus area of cell.

---

# References

## Codes [GitHub Pages]

- AttU-Net: [https://github.com/LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)

- U-Net++: [https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py](https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py)

- MultiResUNet: https://github.com/j-sripad/mulitresunet-pytorch/blob/main/multiresunet.py

- Residual U-Net: https://github.com/rishikksh20/ResUnet

- TransUNet: https://github.com/Beckschen/TransUNet

- UCTransNet: https://github.com/McGregorWwww/UCTransNet

- MISSFormer: https://github.com/ZhifangDeng/MISSFormer

## Other

- 

---

# Citation

```latex

```

---
