
def relu():
    return 'relu'
def conv(param):
    return f"conv: {param}"

params = [1,2,3,4,5]

net_list = [[conv(param), relu()] for param in params]
net_list = [*net_list]
print(*net_list)