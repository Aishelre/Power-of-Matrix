# encoding : UTF-8
# python 3.5

import numpy as np
import numpy.linalg as linalg

class Power_Of_Matrix_Aid():
    def __init__(self):
        self.precision = 3  # 반올림 할 자리수

        self.get_input()
        try:
            self.calculate()
        except np.linalg.linalg.LinAlgError as e:
            print("* Error : ", e)
            print("* 행렬 P의 역행렬이 존재하지 않습니다.")

    def get_input(self):
        """
        사용자로부터 2x2행렬의 원소를 입력받는다. 
        """
        try:
            self.u_input = input("2x2 행렬의 성분 입력. (구분 : ' ') ")
            self.u_input = list(map(float, self.u_input.split(' ')))
            if len(self.u_input) != 4:
                raise IndexError("Too many inputs.")

            self.A = np.array([[self.u_input[0], self.u_input[1]],
                          [self.u_input[2], self.u_input[3]]])

        except (IndexError) as e:
            print("* Error : ", e)
            print("* Input exact 4 numbers.")
            self.get_input()

        except (ValueError) as e:
            print("* Error : ", e)
            print("* You should input Numbers.")
            self.get_input()

    def refine_vector(self, e_value):
        """
        e_value : [2, 1] shape
        This function refines floats of e_value
        ex) [3.2, 4.0] to [4, 5]        
        """
        n1 = e_value[0]
        n2 = e_value[1]
        if n1<0 and n2<0:
            n1, n2 = -n1, -n2
        elif n1==0 and n2==0:
            return np.array([n1, n2])

        while True:
            if n1%1 == 0 and n2%1 == 0:
                break
            else:
                n1 *= 10
                n2 *= 10
        a = np.abs(np.max([n1,n2]))
        b = np.abs(np.min([n1,n2]))
        while True:
            if b == 0:
                break
            a,b = b, a%b
        n1 = n1/a
        n2 = n2/a
        return np.array([n1,n2])

    def calculate(self):
        """
        행렬의 n제곱 계산을 쉽게 하기 위해서
        행렬 A = P M (P^-1) 의 형태로 바꾼다.
        eigenvalue와 eigenvector를 구하고 이를 이용해
        행렬 P, M을 구한다.
        """
        # Calculate eigenValue
        # Var 'term's are for quadratic formula
        term2 = 1
        term1 = -1 * (self.A[0][0] + self.A[1][1])
        term0 = self.A[0][0]*self.A[1][1] - self.A[0][1]*self.A[1][0]

        if term1**2 - 4*term0 < 0:  # 허수부분 존재
            print("* eigenvalue 값에 허수가 존재합니다.\n* 제한 사항에 반하므로 종료합니다.")
            return
        e_value1 = np.round(( -term1 - np.sqrt(term1**2 - 4*term0) ) / 2 * term2, self.precision)  # 근의 공식 이용
        e_value2 = np.round(( -term1 + np.sqrt(term1**2 - 4*term0) ) / 2 * term2, self.precision)  # 근의 공식 이용
        print("eigenValue1 : {}, eigenValue2 : {}".format(e_value1, e_value2))
        if e_value1 == e_value2:
            print("* eigenvalue 값이 1개 존재합니다.\n* 제한 사항에 반하므로 종료합니다.")
            return

        # Calculate eigenVector
        e_vector1 = np.array([-self.A[0][1], self.A[0][0] - e_value1])
        e_vector2 = np.array([-self.A[0][1], self.A[0][0] - e_value2])
        e_vector1 = self.refine_vector(e_vector1)
        e_vector2 = self.refine_vector(e_vector2)
        print("eigenVector1 : {}, eigenVector2 : {}".format(e_vector1, e_vector2))

        self.P = np.transpose(np.array([[e_vector1], [e_vector2]])).reshape((2,2))
        self.M = np.array([[e_value1,0], [0, e_value2]]).reshape((2,2))
        self.P_inv = linalg.inv(self.P)
        print("===============================")
        print("Alpha : {}, Beta : {}".format(e_value1, e_value2))
        print("Matrix P : \n", self.P)
        print("Matrix M : \n", self.M)
        print("Matrix P_inverse : \n",self.P_inv)

Power_Of_Matrix_Aid()