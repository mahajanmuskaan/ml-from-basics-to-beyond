# Day 5: Python Basics for Machine Learning
# Topics: Variables, Loops, Functions


# Variables
x = 20
y = 30
print("Value of x:", x)
print("Value of y:", y)

print("Sum of x and y:", x+y)

#Incrementing value of x by 10 and Decrementing value of y by 10
x += 10
y -= 10

print("Value of x:", x)
print("Value of y:", y)

# Loops
print("Using for loop to print numbers from 1 to 5:")
for i in range(1,6):
    print(i)
print("Using while loop to print numbers from 5 to 1:")
count = 5
while count>0:
    print(count)
    count -= 1


# Functions
#function to add 2 numbers
def add_numbers(a,b):
    return a+b

#Take input from users to add 2 numbers
num1 = int(input("Enter first number to add: "))
num2 = int(input("Enter second number to add: "))
result = add_numbers(num1,num2)
print("Sum of", num1, "and", num2, "is:", result)