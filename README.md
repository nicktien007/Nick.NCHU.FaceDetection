# 物聯網應用與資料分析 Final Project

> Python 3 利用 Dlib 實現人臉識別門禁系統

## Steps
依賴套件
```
pip3 install opencv-python
pip3 install scikit-image
pip3 install dlib
```

### 圖片辦識

![](https://i.imgur.com/01GClOE.png)

參數

`照片路徑`

```python
python3 img_rec.py ./test/Zhiling_1.jpg
```

### SHOW 臉部68個特徵點
![](https://i.imgur.com/dzGGdzv.jpg)

```python
python3 img_rec.py ./test/Zhiling_1.jpg
```

### 啟用臉部識別門禁系統

未登錄的User, 顯示`Uknown`，系統鎖定

可點擊鍵盤「**S**」鍵 ，進行人臉登錄

![](https://i.imgur.com/o2uHOug.jpg)



辦識成功並解鎖系統

![](https://i.imgur.com/cqWEWr7.jpg)

## Demo 
[![](http://img.youtube.com/vi/F040UelUwVU/0.jpg)](http://www.youtube.com/watch?v=F040UelUwVU "")


## Refs

- https://bit.ly/3hi4Hmd
- https://bit.ly/34Ipjis
- https://bit.ly/3aWhcCT
- https://bit.ly/3hqVacr
- https://bit.ly/3nRSl6J