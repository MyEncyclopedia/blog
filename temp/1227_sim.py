import random

def run(N: int) -> bool:
	pos = [i for i in range(N)]

	def swap(i, j):
		tmp = pos[i]
		pos[i] = pos[j]
		pos[j] = tmp

	for i in range(N-1):
		if i == 0:
			rnd = random.randint(0, N-1)
			print(f'{i} choose1 {pos[rnd]}')
			swap(rnd, 0)
		else:
			if pos[i] == i:
				pass
			else:
				# remaining
				rnd = random.randint(i, N-1)
				print(f'{i} choose3 {pos[rnd]}')
				swap(rnd, i)
		print(f'{i} => {pos}')
	return pos[N-1] == N-1


if __name__ == "__main__":
	N = 5
	ret = {True: 0, False: 0}
	for _ in range(1000):
		result = run(N)
		print(f'{result}\n')
		ret[result] += 1

	print(f'{ret}')


