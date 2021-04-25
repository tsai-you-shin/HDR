# HDR
High Dynamic Range Imaging

- Participants: 童筱妍 蔡宥杏
- [Report](https://hackmd.io/@IYvh1Iq5QwSChHCr06GTQA/HyTUz9Ew8) on Hackmd, click to see more details

## 什麼是HDR
High Dynamic Range Imaging是用來實現比普通點陣圖像技術(常受限於明暗差距)更大曝光動態範圍的一種技術，使得圖片能顯示出更大的明暗變化，讓圖片經由計算重整能更接近人眼所見的世界。
- 整體實作過程可分為三個部分:
    - 影像對齊(image alignment)
    - 藉由不同曝光時間的照片，回推計算真實場景的能量分布(LDR->HDR)
    - 經由tone mapping將場景能量轉換成普通螢幕能顯示的影像(HDR->LDR)


