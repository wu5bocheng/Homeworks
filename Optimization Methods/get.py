import requests
url = "http://i.pximg.net/img-master/img/2021/10/10/21/08/32/93358924_p0_master1200.jpg"
headers = {
    "referer":"https://www.pixiv.net/",
}

a = requests.get(url = url,headers = headers)
with open('1.jpg','wb') as file:  # 以byte形式将图片数据写入
    file.write(a.content)