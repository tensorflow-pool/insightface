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

# print("------------------------------")
# x = nd.ones((2, 3)) * 2
# x.attach_grad()
# with autograd.record():
#     t4 = nd.log(x)
# print("t4", t4)
# t4.backward()
#
# print("x", x.grad)

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

#
# print("------------------------------")
# x = nd.ones((4, 3)) * 2
# x[0, 0] = 4
# x.attach_grad()
# with autograd.record():
#     t = nd.SoftmaxOutput(x, nd.array([1, 1, 1, 2]), normalization='valid', use_ignore=True, ignore_label=2)
#     # t1 = nd.exp(x)
#     # # print("t1", t1)
#     # # t1.attach_grad()
#     # t2 = nd.broadcast_div(t1, nd.sum(t1, axis=-1, keepdims=True))
#     # t2 = nd.broadcast_mul(t2, nd.array([1, 1, 1, 0]).expand_dims(axis=1))
#     # print("t2", t2)
#     # # t2.attach_grad()
#     # t = -nd.pick(t2, nd.array([1, 1, 1, 2]), axis=1)
# print("t", t)
# t.backward()
#
# print("x", x.grad)
# # print("t1", t1.grad)
# # print("t2", t2.grad)


# print("------------------------------")
# x = nd.ones((4, 3)) * 2
# x[0, 0] = 64
# x[0, 1] = -64
# x[3, 0] = 64
# x[3, 1] = -64
# print(x)
# x.attach_grad()
# with autograd.record():
#     t1 = nd.exp(x)
#     print("t1", t1)
#     # t1.attach_grad()
#     t2 = nd.broadcast_div(t1, nd.sum(t1, axis=-1, keepdims=True))
#     t2 = nd.clip(t2, a_min=1.18e-38, a_max=3.4e38)
#     print("t2", t2)
#     t3 = nd.log(t2)
#     print("t3", t3)
#     t = -nd.sum(t3, axis=-1)
#
#     # t = nd.broadcast_mul(t, nd.array([1, 1, 1, 0]))
# print("t", t)
# t.backward()
#
# print("x", x.grad)
# # print("t1", t1.grad)
# # print("t2", t2.grad)


print("------------------------------")
x = nd.ones((4, 3)) * 2
x[0, 0] = 64
x[0, 1] = -64
x[3, 0] = 64
x[3, 1] = -64
print("x", x)
x.attach_grad()
with autograd.record():
    # t = nd.SoftmaxOutput(x, nd.array([1, 1, 1, 2]), normalization='valid', use_ignore=False, ignore_label=2)
    t1 = nd.exp(x)
    # print("t1", t1)
    t1.attach_grad()
    t2 = nd.broadcast_div(t1, nd.sum(t1, axis=-1, keepdims=True))
    t2 = nd.clip(t2, a_min=1.18e-38, a_max=3.4e38)
    t2.attach_grad()
    print("t2", t2)
    # t2.attach_grad()
    t = nd.pick(t2, nd.array([1, 1, 1, 2]), axis=1)
    print("t", t)
    t = -t.log().sum()
print("t", t)
t.backward()

print("x", x.grad)
print("t1", t1.grad)
print("t2", t2.grad)
