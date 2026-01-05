import cv2
from pytesseract import pytesseract as pyt

img = cv2.imread('./ocr1.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
pyt.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 이진화 : 좀 더 정확한 글자 분석
# THRESH_BINARY_INV : 흰 글씨를 검은색으로 반전 (INV 없으면 기본유지)
ret, binary = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

text = pyt.image_to_string(img, lang='eng')
print(text)

# 실습
# 1. 이미지 불러오기
img = cv2.imread('./ocr3.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
pyt.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # 없으면 에러 뜸.

# 2. ROI 지정
h, w = img.shape
roi = img[
    int(h * 0.15):int(h * 0.85),
    int(w * 0.15):int(w * 0.85)
]

# 3. 이진화
ret, binary = cv2.threshold(roi, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 4. OCR 적용
text = pyt.image_to_string(binary, lang='eng')
print(text)

# 확인용 출력
cv2.imshow("ROI", roi)
cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()