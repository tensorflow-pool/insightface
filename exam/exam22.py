from mxnet import nd, autograd

# x = nd.ones((2, 3)) * 2
# x.attach_grad()
# y = nd.ones((2, 3)) * 3
# y.attach_grad()
# z = nd.ones((2, 3)) * 4
# z.attach_grad()
# with autograd.record():
#     t1 = x * y
#     t2 = z * t1
#     # t3 = nd.softmax_cross_entropy(t2, nd.array([1, 1]))
#     # t3 = nd.softmax(t2)
#     # print("t33", t33)
#     t3 = nd.exp(t2)
#     t3 = nd.broadcast_div(t3, nd.sum(t3, axis=-1, keepdims=True))
#     # print("t3", t3)
#     t3.attach_grad()
#     t4 = nd.log(t3) / 34444444444444444444444444444444444444444444444444444444
#     t4 = t4 * 0.1
# print("t4", t4)
# t4.backward()
#
# print("t3g", t3.grad)
# print("t2g", t2.grad)
# print("t1g", t1.grad)
# print("x", x.grad)

print("------------------------------")
x = nd.ones((2, 3)) * 2
x.attach_grad()
with autograd.record():
    t4 = nd.log(x)
print("t4", t4)
t4.backward()

print("x", x.grad)

print("------------------------------")
# x = nd.ones((2, 3)) * 2
# x.attach_grad()
# y = nd.ones((2, 3)) * 3
# y.attach_grad()
# z = nd.ones((2, 3)) * 4
# z.attach_grad()
#
# yy = nd.ones((2, 3)) * 3
# yy.attach_grad()
# zz = nd.ones((2, 3)) * 4
# zz.attach_grad()
# with autograd.record():
#     t1 = x * y
#     t2 = z * t1
#
#     tt = x * yy * zz
#
#     ttt = t2 + tt
#
# ttt.backward()
#
# print("t2g", t2.grad)
# print("t1g", t1.grad)
# print("x", x.grad)
