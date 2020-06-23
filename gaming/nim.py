

def canWin(n):
    if n <= 3:
        return True
    for i in range(1, 4):
        if not canWin(n - i):
            return True
    return False

if __name__ == "__main__":
    for i in range(1, 20):
        print(canWin(i))