def dummy():
    return 45

public_data = "public stuff!"
_private_data = "private stuff!"
print('__name__ =', __name__)

# If this code is being *run*, then __name__ will be '__main__'
if __name__ == '__main__':
    # test dummy
    if dummy() == 45:
        print('success')
