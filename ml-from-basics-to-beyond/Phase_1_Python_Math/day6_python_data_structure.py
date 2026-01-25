# Day 6: Python Data Structures for Machine Learning
# Topics: Lists, Tuples, Dictionaries, Sets

# Lists
print("=== Lists ===")
fruits = ["apple", "banana", "cherry"]
print("Original list:", fruits)
fruits.append("date")
print("After appending 'date':", fruits)
fruits.remove("banana")
print("After removing 'banana':", fruits)
print("First fruit:", fruits[0])
print("Second fruit:", fruits[1])
print("Sliced list (first two fruits):", fruits[:2])

# Looping through a list
print("Looping through the list:")
for i in fruits:
    print(i)

# Tuples
print("\n=== Tuples ===")
names = ("Alice", "Bob", "Charlie")
print("Original tuple:", names)
print("First name:", names[0])
print("Second name:", names[1])
print("Sliced tuple (first two names):", names[:2])
# Tuples are immutable, so we cannot add or remove elements
print("Looping through the tuple:")
for name in names:
    print(name)


# Dictionaries
print("\n=== Dictionaries ===")
student = {
    "name": "John",
    "age": 21,
    "courses": ["Math", "Science"]
}
print("Original dictionary:", student)
print("Student name:", student["name"])
student["age"] = 22  # Update age
print("After updating age:", student)
student["grade"] = "A"  # Add new key-value pair
print("After adding grade:", student)
del student["courses"]  # Remove key-value pair
print("After removing courses:", student)
# Looping through a dictionary
print("Looping through the dictionary:")
for key, value in student.items():
    print(f"{key}: {value}")

# Sets
print("\n=== Sets ===")
colors = {"red", "green", "blue"}
print("Original set:", colors)
colors.add("yellow")
print("After adding 'yellow':", colors)
colors.remove("green")
print("After removing 'green':", colors)
# Sets do not support indexing and do not allow duplicate elements
print("Is 'red' in the set?", "red" in colors)
# Looping through a set
print("Looping through the set:")
for color in colors:
    print(color)

