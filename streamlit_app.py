import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("내 남자들 소개")

# 이미지 불러오기
pil_img = Image.open("my_guys.jpg")
tk_img = ImageTk.PhotoImage(pil_img)

label_img = tk.Label(root, image=tk_img)
label_img.pack()

label_text = tk.Label(root, text="이쪽은 내 절친 A, 그리고 여기는 B!", font=("Arial", 14))
label_text.pack()

root.mainloop()