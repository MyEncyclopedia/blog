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
			swap(rnd, N-1)
		else:
			if pos[i] == i:
				print(f'{i} choose2 {i}')
				print('hit')
				swap(i, N-i-1)
			else:
				# remaining
				rnd = random.randint(0, N-i-1)
				print(f'{i} choose3 {pos[rnd]}')
				swap(rnd, N-i-1)
		print(f'{i} => {pos}')
	return pos[0] == N-1


if __name__ == "__main__":
	N = 4
	ret = {True: 0, False: 0}
	for _ in range(500):
		result = run(N)
		# print(result)
		ret[result] += 1

	print(f'{ret}')

	# ret = {0: 0, 1:0, 2:0, 3:0}
	# for _ in range(1000):
	# 	ret[random.randint(0, 3)] += 1
	# print(ret)



