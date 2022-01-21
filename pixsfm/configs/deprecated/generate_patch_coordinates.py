res = []

ps = 4

offset = - ps / 2.0 + 0.5
scale = 1.0
for i in range(ps):
    for j in range(ps):
        res.append([(offset+i)*scale, (offset+j) * scale])
print(res)