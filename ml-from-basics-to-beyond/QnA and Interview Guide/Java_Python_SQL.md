# Java · Python · SQL — Complete Interview Guide
#### 68 Questions · Full Explanations · Code Examples · Conceptual Depth

> 📌 **How to use this guide:**
> Every answer is written to be spoken in an interview — detailed enough to show depth, concise enough to stay focused. Code examples are kept minimal and illustrative. Conceptual diagrams use ASCII for clarity.

---

# PART I — JAVA (20 Questions)

> ☕ **Philosophy:** Java is a class-based, object-oriented, platform-independent language designed with the principle *"Write Once, Run Anywhere."* Every concept in Java traces back to its object model, memory architecture, or its compile-then-interpret execution model.

---

## SECTION A — Java Fundamentals

---

### Q121. Explain the key principles of Object-Oriented Programming in Java.

OOP is built on **four foundational pillars**. Java implements all four, and understanding them is fundamental to understanding the language itself.

---

#### 1. Encapsulation — *"Bundle data and behaviour, hide the internals"*

Encapsulation binds data (fields) and the methods that operate on them into a single unit (class), and restricts direct access to internal state using access modifiers.

```java
public class BankAccount {
    private double balance;      // data hidden from outside

    public void deposit(double amount) {   // controlled access
        if (amount > 0) balance += amount;
    }

    public double getBalance() {           // read-only access
        return balance;
    }
}
```

**Why it matters:** External code cannot directly corrupt `balance`. All modifications go through validated methods. This is the foundation of data integrity.

---

#### 2. Abstraction — *"Show what is necessary, hide what is not"*

Abstraction exposes essential features while hiding implementation complexity. Achieved through **abstract classes** and **interfaces**.

```java
abstract class Shape {
    abstract double area();          // WHAT to do — no HOW

    void display() {                 // shared behaviour
        System.out.println("Area: " + area());
    }
}

class Circle extends Shape {
    double radius;
    Circle(double r) { radius = r; }

    @Override
    double area() { return Math.PI * radius * radius; }  // HOW for Circle
}
```

---

#### 3. Inheritance — *"Build on what already exists"*

A subclass inherits fields and methods from its superclass, enabling code reuse and establishing an **is-a** relationship.

```java
class Animal {
    String name;
    void eat() { System.out.println(name + " eats."); }
}

class Dog extends Animal {
    void bark() { System.out.println(name + " barks."); }
}

// Dog IS-A Animal — it can eat() and bark()
Dog d = new Dog();
d.name = "Tommy";
d.eat();   // inherited
d.bark();  // own method
```

**Key rule:** Java supports **single inheritance** for classes (a class can extend only one class) but **multiple inheritance through interfaces**.

---

#### 4. Polymorphism — *"One interface, many forms"*

The same method name behaves differently depending on the object it is called on. Two types:

- **Compile-time (Static):** Method overloading — resolved by the compiler based on method signature
- **Runtime (Dynamic):** Method overriding — resolved by the JVM based on actual object type

```java
Animal a = new Dog();    // reference type = Animal, object type = Dog
a.eat();                 // calls Dog's overridden eat() at runtime — Dynamic dispatch
```

---

### Q122. What is the difference between JDK, JRE, and JVM?

These three components form Java's execution architecture. Understanding their layered relationship is essential.

```
┌─────────────────────────────────────────────────────────────┐
│                          JDK                                │
│  (Java Development Kit — for DEVELOPERS)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                        JRE                           │   │
│  │  (Java Runtime Environment — for RUNNING programs)   │   │
│  │                                                       │   │
│  │  ┌─────────────────────────────────────────────────┐  │   │
│  │  │                   JVM                           │  │   │
│  │  │  (Java Virtual Machine — executes bytecode)     │  │   │
│  │  │  Class Loader → Bytecode Verifier → Interpreter │  │   │
│  │  └─────────────────────────────────────────────────┘  │   │
│  │  + Java Standard Library (java.lang, java.util, ...)  │   │
│  └──────────────────────────────────────────────────────┘   │
│  + Compiler (javac), Debugger (jdb), javadoc, jar tool      │
└─────────────────────────────────────────────────────────────┘
```

| Component | Full Name | Purpose | Who Needs It |
|---|---|---|---|
| **JVM** | Java Virtual Machine | Executes Java bytecode (.class files). Manages memory, garbage collection, security. Platform-specific. | Runtime |
| **JRE** | Java Runtime Environment | JVM + standard class libraries. Everything needed to run a Java program. | End users running programs |
| **JDK** | Java Development Kit | JRE + compiler (javac) + debugger + tools. Everything needed to write and compile Java. | Developers |

**Key insight:** JVM is platform-specific (a different JVM exists for Windows, Linux, macOS), but Java bytecode is platform-independent — this enables "Write Once, Run Anywhere."

---

### Q123. Explain "Write Once, Run Anywhere" in Java.

**WORA** is Java's core promise — a Java program compiled on any platform can run on any platform with a JVM installed.

```
TRADITIONAL COMPILED LANGUAGE (e.g., C):
  Source code → Compiler → Platform-specific machine code
  Windows .exe → runs only on Windows
  Linux binary  → runs only on Linux

JAVA:
  Source code (.java)
       ↓  javac (compiler)
  Bytecode (.class)          ← platform-independent intermediate code
       ↓  JVM (on any OS)
  Machine code               ← JVM translates for the specific platform

  Same .class file runs on:
    Windows JVM → Windows machine code
    Linux JVM   → Linux machine code
    macOS JVM   → macOS machine code
```

**The mechanism:**
1. `javac Hello.java` → produces `Hello.class` (bytecode — not machine code)
2. `java Hello` → JVM reads `Hello.class`, interprets/compiles to native machine code for the current platform
3. JIT (Just-In-Time) compiler within JVM converts hot bytecode to native code for performance

**Limitation:** The JVM itself is platform-specific (different JVM binaries for different OS). WORA applies to the bytecode, not the JVM.

---

### Q124. What is the difference between primitive types and wrapper classes?

Java has two type systems that serve different purposes.

#### Primitive Types

The 8 built-in value types. Stored directly on the **stack**. No methods. Efficient.

```
byte    (8-bit  integer,  -128 to 127)
short   (16-bit integer,  -32,768 to 32,767)
int     (32-bit integer,  ~ ±2 billion)
long    (64-bit integer,  ~ ±9.2 × 10¹⁸)
float   (32-bit decimal,  ~6-7 significant digits)
double  (64-bit decimal,  ~15 significant digits)
char    (16-bit Unicode,  '\u0000' to '\uFFFF')
boolean (true or false)
```

```java
int x = 42;       // stored as raw value on stack
double pi = 3.14;
```

#### Wrapper Classes

Object versions of primitives in `java.lang`. Stored on the **heap**. Have methods. Required for generics and collections.

```java
Integer a = Integer.valueOf(42);    // object on heap
Double  b = Double.valueOf(3.14);

// Useful methods:
int parsed = Integer.parseInt("123");        // String → int
String s   = Integer.toBinaryString(42);     // "101010"
int max    = Integer.MAX_VALUE;              // 2147483647
```

#### Autoboxing and Unboxing

Java 5+ automatically converts between primitives and wrappers:

```java
Integer obj = 42;        // autoboxing:   int → Integer (compiler adds valueOf)
int prim    = obj;       // unboxing:     Integer → int (compiler adds intValue)

List<Integer> list = new ArrayList<>();
list.add(5);             // autoboxing happens automatically
int val = list.get(0);   // unboxing happens automatically
```

| Property | Primitive | Wrapper |
|---|---|---|
| Memory | Stack (fast) | Heap (slower) |
| Default value | 0 / false | null |
| Usable in collections | ❌ No | ✅ Yes |
| Methods available | ❌ No | ✅ Yes (parseInt, valueOf, etc.) |
| Null possible | ❌ No | ✅ Yes (NullPointerException risk) |
| Performance | Better | Autoboxing overhead |

---

### Q125. Explain pass-by-value in Java. How are objects passed?

**Java is strictly pass-by-value — always.** This is one of the most frequently misunderstood concepts.

**For primitives:** The actual value is copied. The method receives a copy; changes do not affect the original.

```java
void addTen(int x) {
    x = x + 10;    // modifies the LOCAL copy
}

int num = 5;
addTen(num);
System.out.println(num);    // Still 5 — original unchanged
```

**For objects:** The **reference (memory address)** is copied — not the object itself. Both the caller's variable and the method's parameter point to the same object. Modifying the object's state through the reference affects the original.

```java
void changeName(StringBuilder sb) {
    sb.append(" World");    // modifies the OBJECT both references point to
}

StringBuilder str = new StringBuilder("Hello");
changeName(str);
System.out.println(str);    // "Hello World" — object WAS modified

// BUT: reassigning the parameter does NOT affect the caller:
void reassign(StringBuilder sb) {
    sb = new StringBuilder("New");  // sb now points to a different object
}
// After calling reassign(str), str still points to "Hello World"
```

**Mental model:**

```
PRIMITIVE:
  Caller: [num = 5]
  Copy:   [x = 5] → x = 15 (doesn't affect num)

OBJECT:
  Caller: [str → "Hello World" object at 0x1234]
  Copy:   [sb  → same 0x1234]  ← both point to SAME object
  sb.append() → modifies object at 0x1234 → caller's str sees the change
  sb = new    → sb now points to new address → caller's str still at 0x1234
```

---

### Q126. What are access modifiers in Java?

Access modifiers control the **visibility** of classes, methods, and fields from other parts of the program. Java has four levels.

| Modifier | Same Class | Same Package | Subclass (other pkg) | Other Package |
|---|---|---|---|---|
| `public` | ✅ | ✅ | ✅ | ✅ |
| `protected` | ✅ | ✅ | ✅ | ❌ |
| `default` (no keyword) | ✅ | ✅ | ❌ | ❌ |
| `private` | ✅ | ❌ | ❌ | ❌ |

```java
public class AccessDemo {
    public    int a = 1;   // accessible everywhere
    protected int b = 2;   // accessible within package + subclasses
              int c = 3;   // package-private (default) — within package only
    private   int d = 4;   // only within this class
}
```

**Design principle:**
- Fields should almost always be `private` (encapsulation)
- Methods forming the API should be `public`
- Methods shared within a package but not externally: `default`
- Methods for subclasses to override: `protected`

---

## SECTION B — Memory & OOP

---

### Q127. Explain Java's memory model (heap, stack, method area).

The JVM divides memory into distinct regions, each with a specific purpose.

```
JVM MEMORY ARCHITECTURE:

┌────────────────────────────────────────────────────────┐
│                    METHOD AREA                          │
│  (Shared across all threads)                           │
│  Class metadata, static variables, constant pool,      │
│  method bytecode, field/method info                    │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                       HEAP                             │
│  (Shared across all threads)                           │
│  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │  Young Generation │  │     Old Generation        │   │
│  │  Eden + S0 + S1   │  │  (long-lived objects)    │   │
│  └──────────────────┘  └──────────────────────────┘    │
│  All objects created with 'new' live here              │
└────────────────────────────────────────────────────────┘

┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  STACK        │  │  STACK        │  │  STACK        │
│  (Thread 1)   │  │  (Thread 2)   │  │  (Thread 3)   │
│  Frame 1      │  │  Frame 1      │  │  ...          │
│  Frame 2      │  │  ...          │  │               │
│  ...          │  │               │  │               │
│               │  │               │  │               │
│ Each frame:   │  │               │  │               │
│  local vars,  │  │               │  │               │
│  operand stack│  │               │  │               │
│  frame data   │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

**Heap:**
- Stores all **objects** created with `new`
- Shared across all threads (thread-safety concerns arise here)
- Managed by the **Garbage Collector**
- Divided into Young Generation (Eden + Survivor spaces) and Old Generation

**Stack:**
- One stack **per thread** — private, thread-safe
- Stores **stack frames** — one per method call
- Each frame contains: local variables, operand stack, reference to constant pool
- Primitive variables and object references (not objects themselves) live here
- LIFO — automatically reclaimed when method returns (no GC needed)

**Method Area (Metaspace in Java 8+):**
- Stores class-level data: class name, method code, field descriptors, static variables
- Shared across all threads
- In Java 8+, moved to native memory (Metaspace) from PermGen

**Stack Overflow vs OutOfMemoryError:**
- `StackOverflowError` — stack is full (infinite recursion)
- `OutOfMemoryError: Java heap space` — heap is full

---

### Q128. How does garbage collection work in Java?

Garbage Collection (GC) is Java's automatic memory management — it reclaims heap memory occupied by objects that are no longer reachable by any live reference.

**Reachability:** An object is eligible for GC when no active reference chain from any GC root (stack variables, static variables, JNI references) leads to it.

```java
String s = new String("Hello");   // object reachable via s
s = null;                          // reference removed — object NOW eligible for GC
```

#### The Generational GC Model

Based on the empirical observation that **most objects die young** (short-lived temporaries).

```
YOUNG GENERATION (Minor GC — frequent, fast):
  Eden Space:
    New objects allocated here.
    When Eden fills → Minor GC triggered.

  Survivor Space (S0 and S1):
    Objects surviving Minor GC moved to S0.
    Next Minor GC: live objects from Eden + S0 moved to S1, S0 cleared.
    Objects surviving multiple GC cycles (age threshold) → promoted to Old Gen.

OLD GENERATION (Major/Full GC — infrequent, slow):
  Long-lived objects (survived many Minor GCs).
  When Old Gen fills → Full GC triggered (stop-the-world pause).

METASPACE:
  Class metadata. Rarely collected.
```

**GC Algorithms:**

| Algorithm | Description | Best for |
|---|---|---|
| Serial GC | Single-threaded, stop-the-world | Small apps, single-core |
| Parallel GC | Multi-threaded, stop-the-world | Throughput-focused |
| G1 GC (default Java 9+) | Divides heap into regions, concurrent | Balanced latency + throughput |
| ZGC / Shenandoah | Near-zero pause times | Low-latency applications |

**Requesting GC (not guaranteed):**
```java
System.gc();           // suggestion — JVM may or may not comply
Runtime.getRuntime().gc();
```

---

### Q129. What is the difference between == and .equals()?

This is one of the most commonly asked Java questions and the source of many bugs.

**`==` — Reference Equality (identity)**

Checks whether two variables point to the **exact same object in memory** (same heap address). For primitives, compares values directly.

```java
String a = new String("hello");
String b = new String("hello");

System.out.println(a == b);         // FALSE — two different objects in heap
System.out.println(a.equals(b));    // TRUE  — same content
```

**`.equals()` — Logical Equality (content)**

Checks whether two objects are **logically equal** — defined by the class's override of `equals()`. By default (from Object), `equals()` behaves like `==`. Overriding it defines what "equal" means.

```java
// String overrides equals() to compare character sequences
String x = "hello";
String y = "hello";

System.out.println(x == y);       // TRUE (!) — String Pool reuse (same reference)
System.out.println(x.equals(y));  // TRUE — same content
```

**The String Pool gotcha:**
String literals are interned in the String Pool — `"hello"` always returns the same object. But `new String("hello")` always creates a new heap object.

```java
String s1 = "hello";              // in String Pool
String s2 = "hello";              // same pooled object
String s3 = new String("hello");  // new heap object

s1 == s2        → true   (same pooled object)
s1 == s3        → false  (different objects)
s1.equals(s3)   → true   (same content)
```

**Rule:** Use `==` for primitives and intentional reference checks. Use `.equals()` for all object content comparisons. Always check for null before calling `.equals()` — or use `Objects.equals(a, b)` which handles null safely.

---

### Q130. Explain method overloading vs method overriding.

Both relate to polymorphism but operate at different times and serve different purposes.

#### Method Overloading — Compile-Time Polymorphism

**Same method name, different parameter list** within the **same class** (or subclass). Resolved by the compiler based on argument types/count.

```java
class Calculator {
    int add(int a, int b)       { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
    // Return type alone CANNOT distinguish overloads
}

Calculator c = new Calculator();
c.add(2, 3)         → calls first method (int, int)
c.add(2.0, 3.0)     → calls second method (double, double)
c.add(1, 2, 3)      → calls third method (three ints)
```

#### Method Overriding — Runtime Polymorphism

**Same method name, same signature** in a **subclass**, providing a different implementation. Resolved at runtime by the JVM based on the actual object type.

```java
class Animal {
    void sound() { System.out.println("Some sound"); }
}

class Dog extends Animal {
    @Override
    void sound() { System.out.println("Woof"); }  // overrides parent
}

Animal a = new Dog();   // reference: Animal, object: Dog
a.sound();              // Output: "Woof" — Dog's version called at runtime
```

**Rules for overriding:**
- Method signature must be **identical** (name + parameter types)
- Return type must be same or **covariant** (subtype of parent's return type)
- Access modifier can be same or **more permissive** (cannot reduce visibility)
- Cannot override `final`, `static`, or `private` methods
- Use `@Override` annotation — compiler catches mistakes

| Property | Overloading | Overriding |
|---|---|---|
| Where | Same class | Subclass |
| Signature | Different | Identical |
| Resolution | Compile-time | Runtime |
| Polymorphism type | Static | Dynamic |
| Inheritance needed | No | Yes |

---

### Q131. What is polymorphism? Provide examples.

**Polymorphism** (Greek: *many forms*) is the ability of the same interface or method to behave differently depending on the context — the type of object it is called on.

#### Compile-Time Polymorphism (Static Dispatch)

Resolved by the compiler. Achieved through **method overloading** and **operator overloading** (Java doesn't support operator overloading, only method overloading).

```java
class Printer {
    void print(int x)    { System.out.println("Integer: " + x); }
    void print(String x) { System.out.println("String: "  + x); }
    void print(double x) { System.out.println("Double: "  + x); }
}

Printer p = new Printer();
p.print(42);        // compiler selects print(int)
p.print("Hello");   // compiler selects print(String)
p.print(3.14);      // compiler selects print(double)
```

#### Runtime Polymorphism (Dynamic Dispatch)

Resolved by the JVM at runtime. Achieved through **method overriding** and **interface implementations**.

```java
abstract class Shape {
    abstract double area();
    void describe() {
        System.out.println("Area = " + area());
    }
}

class Circle    extends Shape { double area() { return Math.PI * r * r; } }
class Rectangle extends Shape { double area() { return w * h; }          }
class Triangle  extends Shape { double area() { return 0.5 * b * h; }    }

// Polymorphic array — one type, many behaviours:
Shape[] shapes = { new Circle(5), new Rectangle(4,6), new Triangle(3,8) };

for (Shape s : shapes) {
    s.describe();   // JVM decides WHICH area() to call at runtime
}
// Output:
// Area = 78.54
// Area = 24.0
// Area = 12.0
```

**Interface-based polymorphism:**

```java
interface Drawable {
    void draw();
}

class Circle    implements Drawable { public void draw() { System.out.println("Drawing Circle");    } }
class Rectangle implements Drawable { public void draw() { System.out.println("Drawing Rectangle"); } }

List<Drawable> items = List.of(new Circle(), new Rectangle());
items.forEach(Drawable::draw);   // different draw() called for each
```

**Significance:** Polymorphism enables writing code that works with the abstract type (`Shape`, `Drawable`) and automatically handles all concrete subtypes — including future ones not yet written. This is the open/closed principle.

---

## SECTION C — Collections & Data Structures

---

### Q132. Compare ArrayList vs LinkedList. When would you use each?

Both implement the `List` interface but use fundamentally different internal data structures.

#### ArrayList — Dynamic Array

```
Internal structure:
[E0][E1][E2][E3][E4][  ][  ][  ]   ← contiguous memory block
 0   1   2   3   4
```

- Backed by a resizable array. Default initial capacity = 10. When full, grows by 50% (new array of 1.5× size, all elements copied).
- **Random access:** O(1) — direct index calculation
- **Search:** O(n) — linear scan (or O(log n) if sorted + binary search)
- **Insert/delete at end:** Amortised O(1)
- **Insert/delete in middle:** O(n) — elements must shift

#### LinkedList — Doubly Linked List

```
Internal structure:
null ← [prev|E0|next] ↔ [prev|E1|next] ↔ [prev|E2|next] → null
          head                                 tail
```

- Each element is a `Node` object containing data, a reference to the previous node, and a reference to the next node.
- **Random access:** O(n) — must traverse from head or tail
- **Search:** O(n)
- **Insert/delete at head/tail:** O(1) — just update pointers
- **Insert/delete at middle:** O(n) for finding position, O(1) for the pointer update

#### Comparison Table

| Operation | ArrayList | LinkedList |
|---|---|---|
| `get(index)` | **O(1)** | O(n) |
| `add(element)` at end | **Amortised O(1)** | O(1) |
| `add(index, element)` | O(n) | O(n) to find + O(1) to insert |
| `remove(index)` | O(n) | O(n) to find + O(1) to remove |
| `remove(head)` | O(n) (shift) | **O(1)** |
| Memory per element | Less (primitive array) | More (Node object overhead) |
| Cache performance | **Better** (contiguous) | Worse (nodes scattered in heap) |

#### Decision Guide

```
Use ArrayList when:
  ✅ Frequent random access by index (get/set)
  ✅ Iterating through all elements
  ✅ Adding mostly to the end
  ✅ Memory efficiency matters
  ✅ 90% of real-world use cases

Use LinkedList when:
  ✅ Frequent insertions/deletions at BOTH ends (queue, deque)
  ✅ Implementing Stack or Queue (use ArrayDeque instead — even better)
  ✅ You need ListIterator.add/remove frequently in the middle during iteration
```

**Practical note:** `ArrayDeque` outperforms `LinkedList` for most queue/stack use cases in Java because of better cache locality.

---

### Q133. Explain the internal working of HashMap.

HashMap is Java's most important data structure — a hash table implementing the `Map` interface. Understanding its internals is critical.

#### Core Structure

```
HashMap Internal (Java 8+):

Array of Buckets (default capacity = 16):
  [0] → null
  [1] → Node("key1", val1) → Node("key5", val5)  ← collision chain
  [2] → null
  [3] → Node("key2", val2)
  [4] → null
  ...
  [15]→ Node("key3", val3)

When a bucket's chain > 8 nodes (TREEIFY_THRESHOLD):
  Chain converts to a Red-Black Tree for O(log n) worst case
```

#### How `put(key, value)` Works

```
Step 1: Compute hash
  int hash = key.hashCode();
  // Spread high bits to reduce collisions:
  hash = hash ^ (hash >>> 16);

Step 2: Compute bucket index
  int index = hash & (capacity - 1);
  // For capacity=16: index = hash & 15 → always 0-15

Step 3: Check bucket
  If bucket empty → create new Node(hash, key, value) → store in bucket

  If bucket occupied (collision):
    Traverse chain/tree
    If key found (equals check): UPDATE the value
    If key not found: APPEND new Node to chain (or insert in tree)

Step 4: Check load factor
  loadFactor = size / capacity
  Default load factor = 0.75
  If loadFactor > 0.75 → RESIZE (double capacity, rehash all entries)
```

#### How `get(key)` Works

```
Step 1: Compute hash → compute bucket index (same as put)
Step 2: Go to bucket
Step 3: If first node matches key → return value
Step 4: If chain/tree → traverse until key found or null
Time: O(1) average, O(log n) worst case (with tree), O(n) old Java versions
```

#### Why hashCode() and equals() Must Be Consistent

```java
// If two objects are equal (equals() = true),
// they MUST have the same hashCode.
// If not: get() will look in wrong bucket → never finds key put in.

class BadKey {
    int id;
    @Override
    public boolean equals(Object o) { ... }
    // Forgot to override hashCode()!
    // Two "equal" objects can go to different buckets → HashMap breaks.
}
```

**The contract:** `a.equals(b) == true` → `a.hashCode() == b.hashCode()` (mandatory). Reverse not required.

---

### Q134. What is the difference between HashMap and Hashtable?

Both implement the `Map` interface as hash tables, but they differ in thread-safety, null handling, and modern usability.

| Property | HashMap | Hashtable |
|---|---|---|
| **Thread Safety** | ❌ Not synchronized | ✅ All methods synchronized |
| **Null keys** | ✅ Allows ONE null key | ❌ Throws NullPointerException |
| **Null values** | ✅ Allows multiple null values | ❌ Throws NullPointerException |
| **Performance** | **Faster** (no synchronisation overhead) | Slower (every method locks) |
| **Iteration** | `Iterator` (fail-fast) | `Enumeration` (legacy) + `Iterator` |
| **Legacy?** | Modern (Java 2+) | Legacy (Java 1.0) — avoid in new code |
| **Inheritance** | `AbstractMap` | `Dictionary` (obsolete class) |

**Modern replacement for thread-safe use:**

```java
// Option 1: Collections.synchronizedMap (wraps HashMap)
Map<K,V> syncMap = Collections.synchronizedMap(new HashMap<>());

// Option 2: ConcurrentHashMap (better — segment-level locking)
Map<K,V> concMap = new ConcurrentHashMap<>();
// ConcurrentHashMap allows concurrent reads without locking,
// and segment-level locking for writes — much better than Hashtable's
// coarse-grained method-level locking
```

**Interview answer in one line:** "Hashtable is the legacy thread-safe version; HashMap is faster and modern. For thread safety today, use `ConcurrentHashMap` instead of either."

---

### Q135. Compare HashSet, LinkedHashSet, and TreeSet.

All implement the `Set` interface — no duplicate elements. They differ in ordering and performance.

```
SET HIERARCHY:
  Set<E>
    ├── HashSet<E>           — No ordering guarantee, O(1) operations
    ├── LinkedHashSet<E>     — Insertion-order maintained, O(1) operations
    └── SortedSet<E>
          └── TreeSet<E>    — Natural/custom sorted order, O(log n) operations
```

#### HashSet

- Backed by a `HashMap` (elements are keys, dummy value is value)
- No guaranteed iteration order
- O(1) average for add, remove, contains
- Allows one `null` element

```java
Set<String> hash = new HashSet<>();
hash.add("banana"); hash.add("apple"); hash.add("cherry");
System.out.println(hash);  // [banana, cherry, apple] — arbitrary order
```

#### LinkedHashSet

- Backed by a `LinkedHashMap` — maintains a doubly linked list alongside the hash table
- **Iteration order = insertion order**
- Slightly slower than HashSet (linked list maintenance)

```java
Set<String> linked = new LinkedHashSet<>();
linked.add("banana"); linked.add("apple"); linked.add("cherry");
System.out.println(linked);  // [banana, apple, cherry] — insertion order
```

#### TreeSet

- Backed by a `TreeMap` (Red-Black Tree)
- **Elements kept in sorted order** (natural ordering or custom Comparator)
- O(log n) for all operations
- Does NOT allow `null` (can't compare null)

```java
Set<String> tree = new TreeSet<>();
tree.add("banana"); tree.add("apple"); tree.add("cherry");
System.out.println(tree);  // [apple, banana, cherry] — alphabetical

// Custom order:
Set<Integer> desc = new TreeSet<>(Comparator.reverseOrder());
desc.add(3); desc.add(1); desc.add(2);
System.out.println(desc);  // [3, 2, 1]
```

| Property | HashSet | LinkedHashSet | TreeSet |
|---|---|---|---|
| Order | None | Insertion order | Sorted order |
| `add/remove/contains` | O(1) avg | O(1) avg | O(log n) |
| Null elements | One allowed | One allowed | ❌ Not allowed |
| Backed by | HashMap | LinkedHashMap | TreeMap |
| Use when | Max performance | Preserve insertion order | Need sorted iteration |

---

### Q136. What is the Collections Framework in Java?

The Java Collections Framework (JCF) is a unified architecture for storing, manipulating, and accessing groups of objects. It provides a set of **interfaces**, **implementations**, and **algorithms**.

```
COLLECTIONS FRAMEWORK HIERARCHY:

Iterable<E>
  └── Collection<E>
        ├── List<E>           — Ordered, allows duplicates
        │     ├── ArrayList
        │     ├── LinkedList
        │     └── Vector (legacy)
        │
        ├── Set<E>            — No duplicates
        │     ├── HashSet
        │     ├── LinkedHashSet
        │     └── TreeSet (implements SortedSet)
        │
        └── Queue<E>          — FIFO / Priority ordering
              ├── LinkedList
              ├── PriorityQueue
              └── Deque<E>    — Double-ended queue
                    ├── ArrayDeque
                    └── LinkedList

Map<K,V>                      — Key-value pairs (NOT extends Collection)
  ├── HashMap
  ├── LinkedHashMap
  ├── TreeMap (implements SortedMap)
  ├── Hashtable (legacy)
  └── ConcurrentHashMap
```

**The `Collections` utility class** (note: lowercase s) provides static algorithms:

```java
List<Integer> list = Arrays.asList(3, 1, 4, 1, 5, 9);

Collections.sort(list);                        // sort
Collections.reverse(list);                     // reverse
Collections.shuffle(list);                     // shuffle randomly
int max = Collections.max(list);               // find max
Collections.frequency(list, 1);               // count occurrences
List<Integer> synced = Collections.synchronizedList(list); // thread-safe wrapper
List<Integer> unmod  = Collections.unmodifiableList(list); // immutable view
```

---

## SECTION D — Advanced Java

---

### Q137. What are generics in Java? Why use them?

Generics (introduced in Java 5) allow classes, interfaces, and methods to operate on **parameterised types** — types that are specified when the class is used, not when it is defined.

**Problem without generics (pre-Java 5):**

```java
List list = new ArrayList();
list.add("Hello");
list.add(42);              // compiles — no type checking!

String s = (String) list.get(0);   // manual cast needed
String t = (String) list.get(1);   // ClassCastException at RUNTIME!
```

**With generics:**

```java
List<String> list = new ArrayList<>();
list.add("Hello");
list.add(42);              // COMPILER ERROR — caught at compile time ✅

String s = list.get(0);   // no cast needed — compiler knows the type
```

**Generic class:**

```java
public class Pair<T, U> {
    private T first;
    private U second;

    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }

    public T getFirst()  { return first;  }
    public U getSecond() { return second; }
}

Pair<String, Integer> p = new Pair<>("Age", 25);
String key   = p.getFirst();   // returns String — no cast
Integer val  = p.getSecond();  // returns Integer — no cast
```

**Bounded type parameters:**

```java
// T must be a Number or subclass of Number
public <T extends Number> double sum(List<T> list) {
    return list.stream().mapToDouble(Number::doubleValue).sum();
}

sum(List.of(1, 2, 3));        // works — Integer extends Number
sum(List.of(1.5, 2.5));       // works — Double extends Number
// sum(List.of("a", "b"));    // compile error — String is not a Number
```

**Wildcards:**

```java
List<? extends Number>  readOnly  = ...  // read elements as Number, cannot add
List<? super Integer>   writeList = ...  // can add Integer or subtypes
List<?>                 unknown   = ...  // completely unknown type
```

**Why use generics:**
1. **Type safety** — errors caught at compile time, not runtime
2. **Eliminate casts** — cleaner code
3. **Enable reusable algorithms** — write once, works for any type
4. **Type erasure** — generic type info removed at runtime for backward compatibility (implementation detail)

---

### Q138. Explain lambda expressions in Java 8+.

Lambda expressions provide a concise way to represent **anonymous function implementations of functional interfaces** (interfaces with exactly one abstract method).

**Before lambdas (anonymous inner class):**

```java
List<String> names = Arrays.asList("Charlie", "Alice", "Bob");

Collections.sort(names, new Comparator<String>() {
    @Override
    public int compare(String a, String b) {
        return a.compareTo(b);
    }
});
```

**With lambda:**

```java
Collections.sort(names, (a, b) -> a.compareTo(b));
// Or even more concise:
names.sort(String::compareTo);   // method reference
```

**Lambda syntax:**

```
(parameters) -> expression
(parameters) -> { statements; }

// No parameters:
() -> System.out.println("Hello")

// One parameter (parentheses optional):
x -> x * x

// Multiple parameters:
(x, y) -> x + y

// Multiple statements:
(x, y) -> {
    int sum = x + y;
    return sum;
}
```

**Functional interfaces:**

```java
// Predicate<T>: T → boolean
Predicate<String> isLong = s -> s.length() > 5;
isLong.test("Hello")     → false
isLong.test("Welcome!")  → true

// Function<T,R>: T → R
Function<String, Integer> length = String::length;
length.apply("Hello")   → 5

// Consumer<T>: T → void
Consumer<String> print = System.out::println;
print.accept("Hello")   → prints "Hello"

// Supplier<T>: () → T
Supplier<List<String>> listFactory = ArrayList::new;
List<String> l = listFactory.get();

// BiFunction<T,U,R>: T,U → R
BiFunction<Integer,Integer,Integer> add = (a,b) -> a+b;
add.apply(3,4)  → 7
```

**Lambdas and the Stream API:**

```java
List<String> names = List.of("Alice", "Bob", "Charlie", "David");

names.stream()
     .filter(n -> n.length() > 4)    // Predicate
     .map(String::toUpperCase)        // Function
     .sorted()                        // natural order
     .forEach(System.out::println);   // Consumer

// Output:
// ALICE
// CHARLIE
// DAVID
```

**Variable capture:** Lambdas can capture local variables from enclosing scope, but those variables must be effectively final (assigned only once).

---

### Q139. What is the Stream API? How does it differ from loops?

The Stream API (Java 8+) provides a **declarative, functional approach** to processing sequences of elements. A Stream is a pipeline of operations on a data source.

**Key distinction:**
- **Loops (imperative):** You describe *HOW* to do it — step by step
- **Streams (declarative):** You describe *WHAT* you want — the runtime decides how

```java
List<Integer> numbers = List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// IMPERATIVE (loop):
List<Integer> result = new ArrayList<>();
for (int n : numbers) {
    if (n % 2 == 0) {
        result.add(n * n);
    }
}
Collections.sort(result);

// DECLARATIVE (stream):
List<Integer> result = numbers.stream()
    .filter(n -> n % 2 == 0)   // intermediate
    .map(n -> n * n)            // intermediate
    .sorted()                   // intermediate
    .collect(Collectors.toList()); // terminal
```

#### Stream Pipeline Structure

```
DATA SOURCE      INTERMEDIATE OPERATIONS      TERMINAL OPERATION
(collection,     (lazy — not executed until    (triggers execution,
 array, etc.)     terminal operation)           produces result)

numbers.stream() → .filter() → .map() → .sorted() → .collect()
                   LAZY        LAZY       LAZY        TRIGGERS ALL
```

**Key operations:**

```java
// INTERMEDIATE (return Stream<T>):
.filter(predicate)           // keep elements matching predicate
.map(function)               // transform each element
.flatMap(function)           // flatten nested collections
.sorted()                    // natural order
.sorted(comparator)          // custom order
.distinct()                  // remove duplicates
.limit(n)                    // take first n elements
.skip(n)                     // skip first n elements
.peek(consumer)              // debug — see elements without consuming

// TERMINAL (return result or void):
.collect(Collectors.toList())    // → List
.collect(Collectors.toSet())     // → Set
.collect(Collectors.joining(",")) // → String
.count()                          // → long
.findFirst()                      // → Optional<T>
.reduce(identity, accumulator)    // → single value
.forEach(consumer)               // → void
.anyMatch(predicate)             // → boolean
.allMatch(predicate)             // → boolean
.min(comparator)                 // → Optional<T>
.max(comparator)                 // → Optional<T>
```

**Parallel streams:**

```java
// Automatically splits work across multiple CPU cores:
numbers.parallelStream()
       .filter(n -> n % 2 == 0)
       .map(n -> n * n)
       .collect(Collectors.toList());
// Order not guaranteed with parallel streams
```

**Streams vs Loops:**

| Property | Loop | Stream |
|---|---|---|
| Style | Imperative | Declarative |
| Readability | Lower for complex logic | Higher |
| Parallelism | Manual (complex) | `.parallelStream()` |
| Lazy evaluation | No | Yes — only processes needed elements |
| Reusability | N/A | Streams cannot be reused (consumed once) |
| Debugging | Easy (print in loop) | Harder (use .peek()) |

---

### Q140. Explain exception handling in Java (try-catch-finally).

Exceptions are events that disrupt normal program flow. Java provides a structured mechanism to detect, handle, and recover from these events.

#### Exception Hierarchy

```
Throwable
  ├── Error               — JVM-level problems (OutOfMemoryError, StackOverflowError)
  │                         SHOULD NOT be caught by application code
  │
  └── Exception
        ├── Checked Exceptions     — Must be caught or declared (throws)
        │     IOException, SQLException, ClassNotFoundException
        │     Compiler FORCES you to handle them
        │
        └── RuntimeException       — Unchecked Exceptions
              NullPointerException, ArrayIndexOutOfBoundsException,
              ClassCastException, ArithmeticException
              Compiler does NOT force handling
```

#### try-catch-finally

```java
public int divide(int a, int b) {
    try {
        // Code that might throw an exception:
        int result = a / b;          // throws ArithmeticException if b=0
        return result;

    } catch (ArithmeticException e) {
        // Handle specific exception:
        System.err.println("Division by zero: " + e.getMessage());
        return -1;

    } catch (Exception e) {
        // Catch-all for other exceptions (catches most specific first!):
        System.err.println("Unexpected error: " + e.getMessage());
        return -1;

    } finally {
        // ALWAYS executes — whether exception occurred or not:
        // Use for cleanup: close files, release connections, unlock resources
        System.out.println("Cleanup code runs here");
    }
}
```

**try-with-resources (Java 7+):**

```java
// Automatically closes resources that implement AutoCloseable:
try (FileReader fr = new FileReader("file.txt");
     BufferedReader br = new BufferedReader(fr)) {

    String line = br.readLine();
    // No need for finally block to close — done automatically

} catch (IOException e) {
    e.printStackTrace();
}
// br and fr are closed automatically here, even if exception occurs
```

#### Checked vs Unchecked

```java
// CHECKED — must handle:
public void readFile(String path) throws IOException {
    FileReader fr = new FileReader(path);    // IOException is checked
    // Must either catch it or declare throws IOException
}

// UNCHECKED — runtime, optional handling:
public int getElement(int[] arr, int i) {
    return arr[i];   // may throw ArrayIndexOutOfBoundsException (unchecked)
    // Compiler doesn't force you to handle it
}
```

#### Custom Exceptions

```java
public class InsufficientFundsException extends Exception {
    private double amount;

    public InsufficientFundsException(double amount) {
        super("Insufficient funds. Short by: " + amount);
        this.amount = amount;
    }

    public double getAmount() { return amount; }
}

// Usage:
public void withdraw(double amount) throws InsufficientFundsException {
    if (amount > balance) {
        throw new InsufficientFundsException(amount - balance);
    }
    balance -= amount;
}
```

**Best practices:**
- Catch the most specific exception first, most general last
- Never catch `Error` (JVM problems)
- Never swallow exceptions with empty catch blocks
- Always use try-with-resources for I/O and database connections
- Log exceptions with meaningful context, not just `e.printStackTrace()`

---

# PART II — PYTHON (20 Questions)

> 🐍 **Philosophy:** Python prioritises readability, simplicity, and productivity. "There should be one obvious way to do it." Python is dynamically typed, interpreted, and uses duck typing — if it walks like a duck and quacks like a duck, it's a duck.

---

## SECTION A — Python Fundamentals

---

### Q141. Explain the Zen of Python and its philosophy.

The Zen of Python (PEP 20) is a collection of 19 aphorisms that guide Python design decisions. Run `import this` in any Python interpreter to see them.

**The most important principles for interview context:**

```
Beautiful is better than ugly.
  → Python code should be aesthetically clean and readable.

Explicit is better than implicit.
  → Don't rely on hidden behaviour; make intent clear.

Simple is better than complex. Complex is better than complicated.
  → Prefer the simplest solution. Add complexity only when needed.

Readability counts.
  → Code is read more than it is written. Optimise for the reader.

Errors should never pass silently. Unless explicitly silenced.
  → Don't ignore exceptions. Handle them or let them propagate explicitly.

There should be one obvious way to do it.
  → Python tries to have one idiomatic way. Contrast with Perl: TIMTOWTDI.

Now is better than never. Although never is often better than right now.
  → Ship working code. But don't rush to the point of introducing bugs.
```

**Practical implications:**

```python
# UN-Pythonic:
if len(items) != 0:          # checks length unnecessarily

# Pythonic:
if items:                    # empty container is falsy

# UN-Pythonic:
for i in range(len(items)):
    print(items[i])

# Pythonic:
for item in items:
    print(item)

# Pythonic with index:
for i, item in enumerate(items):
    print(i, item)
```

---

### Q142. What makes Python an interpreted language?

An **interpreted** language executes code line by line through an interpreter, without a separate explicit compilation step to machine code.

**Python's actual execution process:**

```
Python source (.py)
       ↓  
  CPython compiler
       ↓
  Bytecode (.pyc cached in __pycache__)
       ↓
  CPython Virtual Machine (PVM)
       ↓
  Machine code execution

Note: Python IS compiled — but to bytecode, not native machine code.
The bytecode is then INTERPRETED by the PVM.
```

**Comparison:**

```
C/C++ (compiled):
  source.c → compiler → machine_code.exe → runs directly on CPU
  Fast. Platform-specific. Separate compilation step.

Python (interpreted):
  script.py → CPython → bytecode → PVM → CPU
  Slower. Platform-independent bytecode. Run directly without explicit compile step.

Java (hybrid):
  Source.java → javac → .class (bytecode) → JVM → machine code
```

**Implications of being interpreted:**
- **No separate compile step** — run `python script.py` directly
- **Interactive REPL** — execute one line at a time (`python3` shell)
- **Dynamic typing** — types resolved at runtime, not compile time
- **Slower execution** — PVM overhead vs native machine code
- **Easier debugging** — errors reported with line numbers at runtime
- **Platform independence** — same `.py` runs anywhere CPython is installed

**Alternatives to CPython:**
- **PyPy** — JIT-compiled Python (much faster for CPU-bound tasks)
- **Jython** — Python on JVM
- **IronPython** — Python on .NET

---

### Q143. What is the Global Interpreter Lock (GIL)? How does it affect performance?

The GIL is a **mutex (lock) in CPython** that allows only **one thread to execute Python bytecode at a time**, even on multi-core systems.

```
SINGLE-CORE (no GIL issue):
  Thread 1: ━━━━━━━━━━━━━━━━━━━━━━━━

MULTI-CORE WITHOUT GIL (true parallelism):
  Thread 1: ━━━━━━━━━━━━━━━━━━━━━━━━  (Core 1)
  Thread 2: ━━━━━━━━━━━━━━━━━━━━━━━━  (Core 2)
  Thread 3: ━━━━━━━━━━━━━━━━━━━━━━━━  (Core 3)

MULTI-CORE WITH GIL (CPython reality):
  Thread 1: ━━━━━━ · · · · ━━━━━━ · ·  (only one at a time runs)
  Thread 2: · · · · ━━━━━━ · · ━━━━ ·
  Thread 3: · · · · · · · · ━━━ · · · ·
  (threads take turns — no true parallelism for CPU-bound work)
```

**Why does the GIL exist?**
CPython's memory management (reference counting for garbage collection) is not thread-safe. The GIL prevents two threads from simultaneously modifying an object's reference count, which would cause corruption.

**Impact on performance:**

| Scenario | GIL Impact | Solution |
|---|---|---|
| **CPU-bound** (number crunching, ML training) | **High** — threads can't use multiple cores | Use `multiprocessing` (separate processes, each with own GIL) |
| **I/O-bound** (file I/O, network, DB) | **Low** — GIL released during I/O waits | `threading` works fine — threads yield GIL during I/O |
| **NumPy/Pandas** heavy computation | Low — NumPy releases GIL for C operations | Native C extensions bypass GIL |

```python
# CPU-bound: use multiprocessing
from multiprocessing import Pool

def heavy_compute(x):
    return sum(i*i for i in range(x))

with Pool(4) as p:          # 4 separate processes, each with own GIL
    results = p.map(heavy_compute, [10**6]*4)

# I/O-bound: threading works fine
import threading

def download(url):
    ...  # GIL released during network wait

threads = [threading.Thread(target=download, args=(url,)) for url in urls]
```

**Python 3.13+ news:** Python is working on a "no-GIL" build (PEP 703) — expect the GIL to become optional in future releases.

---

### Q144. Compare lists, tuples, and sets in Python.

These are three of Python's core built-in collection types, each with distinct characteristics.

#### List — Ordered, Mutable, Allows Duplicates

```python
lst = [1, 2, 3, 2, 4]
lst.append(5)           # modify ✅
lst[0] = 10             # reassign ✅
lst.remove(2)           # removes first occurrence
print(lst[0])           # index access ✅
lst.sort()              # in-place sort
```

#### Tuple — Ordered, Immutable, Allows Duplicates

```python
tpl = (1, 2, 3, 2, 4)
# tpl[0] = 10            # TypeError: immutable! ❌
print(tpl[0])           # index access ✅
print(tpl.count(2))     # count occurrences
print(tpl.index(3))     # find index

# Tuple packing and unpacking:
point = (3, 4)
x, y = point            # unpacking
a, *rest = (1, 2, 3, 4) # a=1, rest=[2,3,4]

# Single-element tuple requires trailing comma:
single = (42,)          # NOT just (42) which is just parentheses
```

#### Set — Unordered, Mutable, NO Duplicates

```python
s = {1, 2, 3, 2, 4}
print(s)                # {1, 2, 3, 4} — duplicates removed
s.add(5)                # add element
s.remove(1)             # remove (KeyError if absent)
s.discard(99)           # remove (no error if absent)

# Set operations:
a = {1, 2, 3}
b = {2, 3, 4}
a | b                   # union: {1, 2, 3, 4}
a & b                   # intersection: {2, 3}
a - b                   # difference: {1}
a ^ b                   # symmetric difference: {1, 4}
```

#### Comparison Table

| Property | List | Tuple | Set |
|---|---|---|---|
| **Ordered?** | ✅ Yes | ✅ Yes | ❌ No |
| **Mutable?** | ✅ Yes | ❌ No | ✅ Yes |
| **Duplicates?** | ✅ Yes | ✅ Yes | ❌ No |
| **Index access?** | ✅ Yes | ✅ Yes | ❌ No |
| **Hashable?** | ❌ No | ✅ Yes (if all elements hashable) | ❌ No (but frozenset is) |
| **Memory** | More | Less | More (hash table) |
| **Lookup time** | O(n) | O(n) | **O(1)** average |

**When to use:**
- **List:** Ordered sequence that changes (shopping cart, results list)
- **Tuple:** Fixed collection of related values (coordinates, RGB, function returns)
- **Set:** Membership testing, deduplication, mathematical set operations

---

### Q145. How do dictionaries work in Python? What can be keys?

Python's `dict` is a **hash map** — stores key-value pairs with O(1) average lookup.

#### Internal Structure

Similar to Java's HashMap — uses hashing to map keys to bucket indices.

```python
d = {'name': 'Alice', 'age': 25, 'city': 'Mumbai'}

# In CPython (simplified):
# hash('name') → some integer → bucket index
# bucket stores: (hash, key, value)

d['name']      # hash('name') → find bucket → return 'Alice'  O(1)
d['age'] = 26  # hash('age') → find bucket → update value     O(1)
'city' in d    # hash('city') → find bucket → True             O(1)
```

**Python 3.7+:** Dictionaries maintain **insertion order** (language guarantee).

```python
d = {'b': 2, 'a': 1, 'c': 3}
list(d.keys())   # ['b', 'a', 'c'] — insertion order preserved
```

#### What Can Be a Key?

Keys must be **hashable** — they must have a consistent `__hash__()` value (not changing during lifetime) and support `__eq__()`.

```python
# VALID KEYS (hashable):
d = {
    'string': 1,       # str ✅
    42: 2,             # int ✅
    3.14: 3,           # float ✅
    True: 4,           # bool ✅
    (1, 2): 5,         # tuple of hashables ✅
    frozenset([1,2]):6,# frozenset ✅
}

# INVALID KEYS (unhashable):
# d[['a','b']] = 1     # list ❌ — TypeError: unhashable type: 'list'
# d[{'x': 1}] = 2      # dict ❌ — unhashable
# d[{1,2,3}] = 3       # set  ❌ — unhashable
```

**Key methods:**

```python
d = {'a': 1, 'b': 2, 'c': 3}

d.keys()           # dict_keys(['a', 'b', 'c'])
d.values()         # dict_values([1, 2, 3])
d.items()          # dict_items([('a',1),('b',2),('c',3)])

d.get('a')         # 1 (safe — returns None if missing)
d.get('z', 0)      # 0 (default value if key missing)

d.setdefault('d', 4)    # inserts 'd':4 if not present, returns value
d.update({'e': 5})      # merge another dict

d.pop('a')         # remove and return value
d.popitem()        # remove and return last inserted item

# Dictionary comprehension:
squares = {x: x**2 for x in range(1, 6)}
# {1:1, 2:4, 3:9, 4:16, 5:25}
```

---

### Q146. Explain mutable vs immutable objects.

**Mutability** determines whether an object's state can be changed after creation.

#### Immutable Objects

Cannot be changed after creation. Any "modification" creates a **new object**.

```python
s = "hello"
id_before = id(s)
s = s + " world"   # creates NEW string object
id_after = id(s)
print(id_before == id_after)  # False — different object!

# Immutable types: int, float, complex, bool, str, tuple, frozenset, bytes
x = 42
x += 1             # x now refers to a NEW int object (43), not modified 42
```

#### Mutable Objects

Can be changed in-place. Same object, different state.

```python
lst = [1, 2, 3]
id_before = id(lst)
lst.append(4)       # MODIFIES the existing list
id_after = id(lst)
print(id_before == id_after)  # True — same object!

# Mutable types: list, dict, set, bytearray, most user-defined classes
```

#### Why This Matters — Shared References

```python
# IMMUTABLE — no aliasing issue:
a = "hello"
b = a
b = b + "!"      # b gets a NEW string; a unchanged
print(a)         # "hello" — unaffected

# MUTABLE — aliasing issue:
a = [1, 2, 3]
b = a            # b points to THE SAME list
b.append(4)      # modifies the shared list
print(a)         # [1, 2, 3, 4] — a IS affected!

# Fix: make a copy
b = a.copy()     # shallow copy — independent list
b = a[:]         # also shallow copy
import copy
b = copy.deepcopy(a)  # deep copy — fully independent
```

#### Implications

| Aspect | Immutable | Mutable |
|---|---|---|
| **Hashable?** | ✅ Yes (can be dict key, set element) | ❌ No |
| **Thread safety** | ✅ Safe — cannot change | ❌ Need locks |
| **Aliasing risk** | ❌ No risk | ✅ Risk — shared reference |
| **Memory** | New object on every "change" | Efficient in-place modification |

**Why strings are immutable:** Safety (strings used as dict keys must not change hash), memory efficiency (string interning/pooling), and thread safety.

---

## SECTION B — OOP & Functions

---

### Q147. Explain classes and objects in Python.

A **class** is a blueprint defining attributes and behaviours. An **object** is a concrete instance of that blueprint.

```python
class Patient:
    # Class variable — shared by ALL instances
    hospital = "City Hospital"
    patient_count = 0

    # Instance method — 'self' refers to the instance
    def __init__(self, name, age, glucose):
        # Instance variables — unique to each instance
        self.name    = name
        self.age     = age
        self.glucose = glucose
        Patient.patient_count += 1    # update class variable

    def is_diabetic(self):
        return self.glucose > 140

    def __str__(self):
        return f"Patient({self.name}, age={self.age})"

    def __repr__(self):
        return f"Patient(name='{self.name}', age={self.age}, glucose={self.glucose})"

# Create objects (instances):
p1 = Patient("Alice", 45, 165)
p2 = Patient("Bob",   30, 110)

print(p1.name)           # "Alice"
print(p1.is_diabetic())  # True  (165 > 140)
print(p2.is_diabetic())  # False (110 <= 140)
print(Patient.hospital)  # "City Hospital" — class variable
print(Patient.patient_count)  # 2
print(p1)                # "Patient(Alice, age=45)"  — uses __str__
```

#### Inheritance

```python
class DiabetesPatient(Patient):
    def __init__(self, name, age, glucose, insulin):
        super().__init__(name, age, glucose)   # call parent's __init__
        self.insulin = insulin

    def risk_level(self):
        if self.glucose > 180 and self.insulin < 15:
            return "High"
        elif self.glucose > 140:
            return "Medium"
        return "Low"

dp = DiabetesPatient("Priya", 52, 195, 10)
print(dp.is_diabetic())  # True — inherited from Patient
print(dp.risk_level())   # "High"
```

---

### Q148. What is the difference between `__init__` and `__new__`?

These are two stages of Python object creation, often confused.

#### `__new__` — Object Creation

`__new__` is the **constructor** — it creates and returns a new instance of the class. It is a static method that receives the class as its first argument. Called BEFORE `__init__`.

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print(f"__new__ called — creating instance of {cls}")
        instance = super().__new__(cls)   # allocate memory, return instance
        return instance                    # MUST return the new instance

    def __init__(self, value):
        print(f"__init__ called — initialising instance")
        self.value = value

obj = MyClass(42)
# Output:
# __new__ called — creating instance of <class '__main__.MyClass'>
# __init__ called — initialising instance
```

#### `__init__` — Object Initialisation

`__init__` is the **initialiser** — it receives the already-created instance (from `__new__`) and sets up its initial state. It does NOT create the object; it configures it.

```
Creation flow:
  MyClass(42)
      ↓
  __new__(MyClass, 42)  → creates and returns empty instance
      ↓
  __init__(instance, 42) → initialises instance with value=42
      ↓
  returns configured instance
```

#### When to Override `__new__`

Almost never needed in normal classes. Use cases:
1. **Singleton pattern:** Return existing instance if one exists
2. **Immutable type subclassing:** (str, int, tuple) — must customise in `__new__`
3. **Metaclasses**

```python
# Singleton using __new__:
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.data = []

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)   # True — same object
```

---

### Q149. Explain class methods, static methods, and instance methods.

Python has three types of methods that differ in what they receive as their first argument and what they can access.

#### Instance Methods — Most Common

Access the specific instance (`self`). Can access and modify instance state AND class state.

```python
class Temperature:
    unit = "Celsius"   # class variable

    def __init__(self, value):
        self.value = value     # instance variable

    def to_fahrenheit(self):       # instance method
        return self.value * 9/5 + 32

    def describe(self):
        return f"{self.value}° {Temperature.unit}"

t = Temperature(100)
t.to_fahrenheit()   # 212.0
```

#### Class Methods — `@classmethod`

Receive the **class** (`cls`) as first argument, not an instance. Can access and modify class state. Cannot access instance state.

**Common use:** Alternative constructors (factory methods).

```python
class Temperature:
    unit = "Celsius"

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_fahrenheit(cls, f_value):    # alternative constructor
        celsius = (f_value - 32) * 5/9
        return cls(celsius)               # creates new Temperature instance

    @classmethod
    def change_unit(cls, new_unit):       # modify class variable
        cls.unit = new_unit

t = Temperature.from_fahrenheit(212)
print(t.value)   # 100.0

Temperature.change_unit("Kelvin")
print(Temperature.unit)  # "Kelvin"
```

#### Static Methods — `@staticmethod`

Receive neither `self` nor `cls`. Cannot access instance or class state. Just a function that logically belongs to the class's namespace.

```python
class Temperature:
    @staticmethod
    def is_valid_celsius(value):          # utility function, no state needed
        return -273.15 <= value <= 1e6

    @staticmethod
    def celsius_to_kelvin(celsius):
        return celsius + 273.15

Temperature.is_valid_celsius(100)    # True
Temperature.celsius_to_kelvin(0)     # 273.15
```

| Decorator | First Arg | Access | Use For |
|---|---|---|---|
| None | `self` (instance) | Instance + class state | Regular behaviour |
| `@classmethod` | `cls` (class) | Class state only | Alternative constructors, factory methods |
| `@staticmethod` | None | Neither | Utility functions grouped with class |

---

### Q150. What are decorators in Python? How do they work?

A decorator is a **higher-order function** that takes a function as input and returns a modified version of that function — adding behaviour without changing the original code.

**The mechanism:**

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the function")    # added behaviour (before)
        result = func(*args, **kwargs)  # call original function
        print("After the function")     # added behaviour (after)
        return result
    return wrapper

# Using the decorator:
@my_decorator
def greet(name):
    print(f"Hello, {name}!")

# @my_decorator is syntactic sugar for:
# greet = my_decorator(greet)

greet("Alice")
# Output:
# Before the function
# Hello, Alice!
# After the function
```

**Decorator with arguments:**

```python
def repeat(n):                    # outer function takes decorator argument
    def decorator(func):          # middle function takes the function
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

say_hello()
# Hello!
# Hello!
# Hello!
```

**Real-world examples:**

```python
import time
import functools

# Timing decorator:
def timer(func):
    @functools.wraps(func)      # preserves __name__, __doc__ of original
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end   = time.perf_counter()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

slow_function()   # slow_function took 1.0012s

# Python's built-in decorators:
@property          # getter as property
@classmethod       # class method
@staticmethod      # static method
@functools.lru_cache(maxsize=128)   # memoisation cache
@dataclasses.dataclass              # auto-generate __init__, __repr__, __eq__
```

---

### Q151. Explain lambda functions and their limitations.

A lambda function is a **small, anonymous, single-expression function** defined inline using the `lambda` keyword.

```python
# Syntax:
# lambda parameters: expression

# Regular function:
def square(x):
    return x * x

# Lambda equivalent:
square = lambda x: x * x
square(5)   # 25

# Multiple parameters:
add = lambda x, y: x + y
add(3, 4)   # 7

# No parameters:
greet = lambda: "Hello!"

# With default:
power = lambda x, n=2: x ** n
power(3)     # 9
power(3, 3)  # 27
```

**Common uses:**

```python
# Sorting with custom key:
students = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
students.sort(key=lambda s: s[1])          # sort by grade
students.sort(key=lambda s: -s[1])         # sort descending

# With map, filter, reduce:
nums = [1, 2, 3, 4, 5]
squared  = list(map(lambda x: x**2, nums))    # [1,4,9,16,25]
evens    = list(filter(lambda x: x%2==0, nums)) # [2,4]

from functools import reduce
total = reduce(lambda acc, x: acc + x, nums)   # 15
```

**Limitations:**

```python
# 1. SINGLE EXPRESSION ONLY — no statements, no multiple lines:
f = lambda x: if x > 0: x else -x   # SyntaxError
# (use ternary operator instead):
f = lambda x: x if x > 0 else -x    # OK

# 2. NO STATEMENTS — no assignments, no loops, no try-except:
f = lambda x: (result := x*2)        # walrus operator works, but bad practice

# 3. HARDER TO DEBUG — no name in traceback:
# TypeError in lambda shows "<lambda>" not a descriptive name

# 4. POOR READABILITY for complex logic — defeats Python's readability goal:
# Don't use lambda for anything complex. Use a named function.

# 5. CANNOT USE ANNOTATIONS (type hints):
f: Callable[[int], int] = lambda x: x*2   # annotation on assignment OK
# but cannot annotate parameters inside lambda
```

**When NOT to use lambda:**
```python
# Bad — save-to-variable defeats the purpose of anonymous:
f = lambda x: x * 2   # just use def f(x): return x * 2

# Bad — too complex for readability:
result = sorted(data, key=lambda x: (x['dept'], -x['salary'], x['name']))
# Better: define a named function with clear intent
```

---

## SECTION C — Advanced Concepts

---

### Q152. What are generators in Python? How do they differ from lists?

A **generator** is a function that produces a sequence of values **lazily** — one at a time, on demand — using the `yield` keyword instead of `return`.

**The key difference:** A list stores all values in memory. A generator produces values one at a time and discards them — using O(1) memory regardless of sequence length.

```python
# LIST — creates all 1 million integers in memory immediately:
nums_list = [x*x for x in range(1_000_000)]    # ~8 MB in memory

# GENERATOR — produces one value at a time, O(1) memory:
nums_gen = (x*x for x in range(1_000_000))    # just a generator object
```

#### Generator Functions

```python
def fibonacci():
    a, b = 0, 1
    while True:          # infinite sequence!
        yield a          # pause here, return a, resume on next()
        a, b = b, a + b

fib = fibonacci()
next(fib)   # 0
next(fib)   # 1
next(fib)   # 1
next(fib)   # 2
next(fib)   # 3
# ...can run forever, using only O(1) memory

# Use in for loop (calls next() automatically until StopIteration):
for i, f in enumerate(fibonacci()):
    if i >= 10: break
    print(f)    # prints first 10 Fibonacci numbers
```

#### How `yield` Works

```python
def count_up(n):
    print("Start")
    for i in range(n):
        print(f"Before yield {i}")
        yield i                   # PAUSE here, return i to caller
        print(f"After yield {i}") # RESUME here when next() called again
    print("Done")

gen = count_up(3)     # function body does NOT execute yet!
next(gen)             # "Start", "Before yield 0" → returns 0, pauses
next(gen)             # "After yield 0", "Before yield 1" → returns 1, pauses
next(gen)             # "After yield 1", "Before yield 2" → returns 2, pauses
next(gen)             # "After yield 2", "Done" → raises StopIteration
```

#### Generator Expressions

```python
# List comprehension (eager — all computed now):
squares = [x**2 for x in range(100)]      # list, all in memory

# Generator expression (lazy — computed on demand):
squares = (x**2 for x in range(100))      # generator object
list(squares)    # consume: [0, 1, 4, 9, ...]

# Chaining generators (memory-efficient pipeline):
data = (x**2 for x in range(1_000_000))   # step 1
evens = (x for x in data if x % 2 == 0)  # step 2
total = sum(evens)                         # step 3: triggers whole pipeline
# Only one value lives in memory at any step!
```

| Property | List | Generator |
|---|---|---|
| Memory | O(n) — all values stored | O(1) — one value at a time |
| Reusable? | ✅ Yes — iterate multiple times | ❌ No — exhausted after one pass |
| Index access? | ✅ `lst[i]` | ❌ No random access |
| When all needed? | ✅ Better | ❌ Forces materialisation |
| Infinite sequences? | ❌ Cannot | ✅ Yes |
| First value only? | Computes all first | ✅ Stops after first |

---

### Q153. Explain the difference between shallow copy and deep copy.

When you copy a mutable object in Python, you must decide how deeply you want to copy nested objects.

#### Assignment — Not a Copy

```python
a = [1, [2, 3], 4]
b = a            # b is ANOTHER REFERENCE to the SAME list
b.append(99)
print(a)         # [1, [2, 3], 4, 99] — a is affected!
```

#### Shallow Copy — One Level Deep

Creates a new container object, but the **elements are still references** to the same objects.

```python
import copy

original = [1, [2, 3], 4]
shallow  = original.copy()       # or list(original) or original[:]
shallow  = copy.copy(original)   # same thing

# Top-level list is independent:
shallow.append(99)
print(original)    # [1, [2, 3], 4]      — unaffected ✅

# But nested objects are SHARED:
shallow[1].append(99)
print(original)    # [1, [2, 3, 99], 4]  — AFFECTED! ❌
```

```
SHALLOW COPY:
  original → [  1  | ref1 |  4  ]     original list
                       ↓
  shallow  → [  1  | ref1 |  4  ]     new list (different container)
                       ↓
                   [2, 3]              SAME inner list — shared reference!
```

#### Deep Copy — All Levels Independent

Recursively copies all nested objects — completely independent.

```python
original = [1, [2, 3], 4]
deep     = copy.deepcopy(original)

deep[1].append(99)
print(original)    # [1, [2, 3], 4]  — NOT affected ✅
print(deep)        # [1, [2, 3, 99], 4]
```

```
DEEP COPY:
  original → [  1  | ref1 |  4  ]     original list
                       ↓
                   [2, 3]              original inner list

  deep     → [  1  | ref2 |  4  ]     new list
                       ↓
                   [2, 3]              NEW inner list — completely independent!
```

**When to use what:**

```
Same object (assignment):     when you want two names for same thing
Shallow copy:                 when your data has no nested mutable objects,
                              or you want nested objects shared (efficiency)
Deep copy:                    when you need a completely independent copy
                              (serialization, undo/redo, parallel processing)
```

---

### Q154. What are context managers? How do you implement them?

A context manager is a Python object that manages a **setup and teardown** around a block of code — ensuring resources are properly acquired and released.

**The `with` statement:**

```python
# Without context manager (risky):
f = open("data.txt", "r")
data = f.read()
# If exception here, f.close() never called → file handle leak!
f.close()

# With context manager (safe):
with open("data.txt", "r") as f:
    data = f.read()
# f.close() automatically called here — even if exception occurred
```

#### How `with` Works

```
with expression as var:
    block

Equivalent to:
  var = expression.__enter__()
  try:
      block
  except:
      expression.__exit__(exc_type, exc_val, exc_tb)
      raise
  finally:
      expression.__exit__(None, None, None)
```

#### Implementing a Context Manager — Class-Based

```python
class DatabaseConnection:
    def __init__(self, host, db_name):
        self.host    = host
        self.db_name = db_name
        self.conn    = None

    def __enter__(self):
        print(f"Connecting to {self.db_name} at {self.host}")
        self.conn = connect(self.host, self.db_name)  # imaginary
        return self.conn    # value bound to 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.conn.close()
        return False        # False = don't suppress exceptions
        # True = suppress exception (rarely appropriate)

with DatabaseConnection("localhost", "hospital_db") as conn:
    results = conn.query("SELECT * FROM patients")
# Connection automatically closed here
```

#### Implementing with `@contextmanager`

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield                           # code in 'with' block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.3f}s")

with timer("Data processing"):
    # do heavy work here
    result = [x**2 for x in range(10**6)]
# Output: "Data processing: 0.052s"
```

**Common context managers:**

```python
# File I/O:
with open("file.txt") as f: ...

# Thread locking:
import threading
lock = threading.Lock()
with lock:
    # thread-safe code

# Database transaction:
with connection.cursor() as cursor:
    cursor.execute(...)

# Temporary directory:
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # tmpdir automatically deleted on exit

# Suppress specific exceptions:
from contextlib import suppress
with suppress(FileNotFoundError):
    os.remove("maybe_missing.txt")
```

---

### Q155. How does exception handling work in Python?

Python's exception handling uses `try-except-else-finally` blocks.

```
FULL STRUCTURE:

try:
    code that might raise an exception
except SpecificException as e:
    handle specific exception
except (TypeError, ValueError) as e:
    handle multiple exceptions
except Exception as e:
    catch-all for unexpected exceptions
else:
    runs ONLY if no exception was raised (success path)
finally:
    ALWAYS runs — cleanup code
```

```python
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    except TypeError as e:
        print(f"Invalid types: {e}")
        return None
    else:
        print("Division successful")   # only if no exception
        return result
    finally:
        print("Attempt complete")      # always runs

safe_divide(10, 2)
# Division successful
# Attempt complete
# returns 5.0

safe_divide(10, 0)
# Cannot divide by zero
# Attempt complete
# returns None
```

#### Exception Hierarchy

```python
BaseException
  ├── SystemExit           # sys.exit() — don't catch in app code
  ├── KeyboardInterrupt    # Ctrl+C — don't catch in app code
  └── Exception            # catch this or subclasses in application code
        ├── ValueError     # invalid value (int("abc"))
        ├── TypeError      # wrong type
        ├── AttributeError # missing attribute
        ├── KeyError       # missing dict key
        ├── IndexError     # list index out of range
        ├── FileNotFoundError # (subclass of OSError)
        ├── ZeroDivisionError
        └── ...
```

#### Custom Exceptions

```python
class PatientNotFoundError(ValueError):
    def __init__(self, patient_id):
        super().__init__(f"Patient with ID {patient_id} not found")
        self.patient_id = patient_id

def get_patient(pid):
    if pid not in database:
        raise PatientNotFoundError(pid)
    return database[pid]

try:
    patient = get_patient(999)
except PatientNotFoundError as e:
    print(e)              # "Patient with ID 999 not found"
    print(e.patient_id)   # 999

# re-raise with added context:
try:
    result = risky_operation()
except ValueError as e:
    raise RuntimeError("Operation failed during processing") from e
```

---

### Q156. What are list comprehensions? How do they differ from loops?

A list comprehension is a concise, **Pythonic** way to create a list in a single expression.

**Syntax:**

```python
[expression for item in iterable if condition]
```

**Examples:**

```python
# Squares of even numbers 0-19:
squares = [x**2 for x in range(20) if x % 2 == 0]
# [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]

# Equivalent loop:
squares = []
for x in range(20):
    if x % 2 == 0:
        squares.append(x**2)

# Flatten a 2D list:
matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat = [x for row in matrix for x in row]
# [1,2,3,4,5,6,7,8,9]

# String transformation:
names = ["alice", "bob", "charlie"]
upper = [name.upper() for name in names]
# ["ALICE", "BOB", "CHARLIE"]

# Dictionary comprehension:
word_lengths = {word: len(word) for word in names}
# {"alice":5, "bob":3, "charlie":7}

# Set comprehension:
unique_lengths = {len(word) for word in names}
# {3, 5, 7}
```

**List comprehension vs loop:**

| Property | List Comprehension | for Loop |
|---|---|---|
| Speed | ~35-50% faster (optimised C path in CPython) | Slower |
| Readability | Excellent for simple expressions | Better for complex multi-step logic |
| Memory | Creates full list | Can use `append` or generators |
| Scope (Python 3) | Own scope — variables don't leak | Variables leak to enclosing scope |
| Debugging | Harder | Easier (add print statements) |

**When NOT to use list comprehension:**

```python
# Bad — too complex, hurts readability:
result = [transform(x) if condition1(x) else other_transform(x)
          for x in data
          if condition2(x) and condition3(x)]

# Better: use a named function + comprehension, or just a loop
def process(x):
    if condition1(x): return transform(x)
    return other_transform(x)

result = [process(x) for x in data if condition2(x) and condition3(x)]
```

---

## SECTION D — Libraries & Applications

---

### Q157. What Python libraries are essential for data science?

The Python data science ecosystem is built on a layered stack of libraries.

```
DATA SCIENCE PYTHON STACK:

┌────────────────────────────────────────────────────────────┐
│  VISUALISATION                                              │
│  Matplotlib (base) · Seaborn (statistical) · Plotly (interactive) │
├────────────────────────────────────────────────────────────┤
│  MACHINE LEARNING                                           │
│  scikit-learn (classical ML) · XGBoost · LightGBM          │
│  TensorFlow / Keras (deep learning) · PyTorch (deep learning) │
├────────────────────────────────────────────────────────────┤
│  DATA MANIPULATION                                          │
│  pandas (tabular) · NumPy (numerical arrays)               │
├────────────────────────────────────────────────────────────┤
│  STATISTICS & MATH                                          │
│  SciPy · statsmodels                                       │
├────────────────────────────────────────────────────────────┤
│  ENVIRONMENT                                               │
│  Jupyter Notebook/Lab · Anaconda                           │
└────────────────────────────────────────────────────────────┘
```

| Library | Purpose | Key Features |
|---|---|---|
| **NumPy** | Numerical computing | N-dimensional arrays, vectorised operations, C-speed math |
| **pandas** | Data manipulation | DataFrames, missing value handling, groupby, merge/join |
| **Matplotlib** | Base visualisation | Line, bar, scatter, histogram — full control |
| **Seaborn** | Statistical viz | heatmaps, pairplots, built on Matplotlib, beautiful defaults |
| **scikit-learn** | Classical ML | Unified API: fit/predict/transform for all algorithms |
| **SciPy** | Scientific computing | Statistics, optimisation, signal processing, linear algebra |
| **statsmodels** | Statistical modelling | OLS, logistic regression with p-values and confidence intervals |
| **XGBoost/LightGBM** | Gradient boosting | State-of-the-art tabular ML performance |
| **TensorFlow/Keras** | Deep learning | Production deployment, mobile, production |
| **PyTorch** | Deep learning | Research-friendly, dynamic graphs |

---

### Q158. How would you use Python for machine learning?

Python is the dominant language for ML due to scikit-learn's unified API.

**The standard ML workflow in Python:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# 1. LOAD DATA
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. BUILD PIPELINE (prevents data leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(n_jobs=-1, random_state=42))
])

# 4. HYPERPARAMETER TUNING with Cross-Validation
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth':    [5, 10, None]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV F1:",  grid_search.best_score_)

# 5. EVALUATE ON TEST SET (once)
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

### Q159. What is NumPy? Why is it faster than pure Python?

NumPy (Numerical Python) is the foundation of scientific computing in Python — it provides a powerful **N-dimensional array** (`ndarray`) and a comprehensive mathematical function library.

**Why NumPy is fast:**

#### 1. Contiguous Memory (C Arrays)

```
Python list:  [ ptr1 | ptr2 | ptr3 | ptr4 ]   ← pointers to objects scattered in heap
                  ↓       ↓       ↓       ↓
              [int 1] [int 2] [int 3] [int 4]    ← actual values (separate heap objects)

NumPy array:  [ 1 | 2 | 3 | 4 ]               ← actual values side by side in memory
              Single contiguous C array — cache-friendly, no pointer chasing
```

#### 2. Vectorisation — No Python Loop

```python
import numpy as np

n = 1_000_000
lst = list(range(n))
arr = np.arange(n)

# Python loop (slow — each iteration: type check + object overhead):
result = [x * 2 for x in lst]         # ~50ms

# NumPy vectorised (fast — C loop, no Python overhead):
result = arr * 2                        # ~1ms  (~50x faster)
```

#### 3. BLAS/LAPACK for Linear Algebra

NumPy links to optimised BLAS (Basic Linear Algebra Subprograms) libraries for matrix operations — often using CPU SIMD instructions (AVX, SSE) and multi-threading automatically.

```python
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# NumPy matrix multiplication (BLAS-optimised, potentially multi-threaded):
C = A @ B        # ~10ms on modern hardware

# Pure Python triple loop:
# C[i][j] = sum(A[i][k] * B[k][j] for k in range(1000))
# This would take ~300 seconds — 30,000x slower!
```

**Key NumPy concepts:**

```python
# Create arrays:
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones  = np.ones((2, 2))
range_arr = np.arange(0, 10, 0.5)
linspace  = np.linspace(0, 1, 100)   # 100 evenly spaced values

# Shape and type:
arr.shape    # (5,)
arr.dtype    # int64
arr.ndim     # 1

# Indexing and slicing:
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a[1, 2]      # 6 (row 1, col 2)
a[:, 1]      # array([2,5,8]) — entire column 1
a[0:2, 0:2]  # 2x2 subarray

# Broadcasting:
arr = np.array([1, 2, 3])
arr + 10     # [11, 12, 13] — scalar broadcast to array

A = np.ones((3, 3))
v = np.array([1, 2, 3])
A + v        # adds v to each row (broadcasting)
```

---

### Q160. Explain pandas DataFrames and their advantages.

A pandas **DataFrame** is a 2D labelled data structure — like a spreadsheet or SQL table in memory. It is the central data structure for data analysis in Python.

```
DataFrame structure:

         Age   BMI  Glucose  Diabetic
Patient
P001      45  30.1      165         1
P002      32  24.5      110         0
P003      58  33.2      195         1
P004      29  22.8       95         0

  ↑         ↑
Row index   Column labels (named axes)
```

**Creating DataFrames:**

```python
import pandas as pd
import numpy as np

# From dict:
df = pd.DataFrame({
    'Age':      [45, 32, 58, 29],
    'BMI':      [30.1, 24.5, 33.2, 22.8],
    'Glucose':  [165, 110, 195, 95],
    'Diabetic': [1, 0, 1, 0]
})

# From CSV:
df = pd.read_csv('diabetes.csv')
df = pd.read_csv('data.csv', index_col='PatientID', parse_dates=['Date'])
```

**Essential operations:**

```python
# EXPLORATION:
df.shape           # (4, 4)
df.dtypes          # column types
df.info()          # non-null counts + types
df.describe()      # count, mean, std, min, quartiles, max
df.head(5)         # first 5 rows
df.tail(5)         # last 5 rows

# SELECTION:
df['Age']                          # Series (one column)
df[['Age', 'BMI']]                 # DataFrame (multiple columns)
df.loc[0, 'Age']                   # label-based: row 0, col 'Age'
df.iloc[0, 1]                      # position-based: row 0, col 1
df[df['Glucose'] > 140]            # boolean filtering
df.query("Glucose > 140 and BMI > 28")  # SQL-like query

# MISSING VALUES:
df.isnull().sum()                  # count per column
df.dropna()                        # drop rows with any NaN
df.fillna(df.mean())               # fill with column mean
df['BMI'].fillna(df['BMI'].median(), inplace=True)

# TRANSFORMATION:
df['BMI_category'] = pd.cut(df['BMI'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight','Normal','Overweight','Obese'])
df['log_glucose'] = np.log1p(df['Glucose'])

# GROUPBY:
df.groupby('Diabetic')['Glucose'].mean()
df.groupby('Diabetic').agg({'Glucose':'mean', 'BMI':'median'})

# MERGE/JOIN:
result = pd.merge(df_patients, df_diagnoses,
                  on='PatientID', how='left')

# SORTING:
df.sort_values('Glucose', ascending=False)
df.sort_values(['Diabetic', 'Age'])

# APPLY:
df['Risk'] = df['Glucose'].apply(lambda g: 'High' if g > 140 else 'Low')

# PIVOT:
pivot = df.pivot_table(values='Glucose', index='Diabetic',
                        columns='BMI_category', aggfunc='mean')
```

**Advantages over pure Python lists/dicts:**

| Advantage | Detail |
|---|---|
| **Labelled axes** | Access by column name, not index number |
| **Vectorised operations** | NumPy-powered — fast even on millions of rows |
| **Missing value handling** | NaN support throughout all operations |
| **I/O** | Read/write CSV, Excel, SQL, JSON, Parquet natively |
| **GroupBy** | SQL-style aggregations in one line |
| **Merge/Join** | SQL-style table joining |
| **Time series** | DatetimeIndex, resample, rolling windows built-in |
| **Integration** | Native integration with matplotlib, scikit-learn, SQL |

---

# PART III — SQL (28 Questions)

> 🗄️ **Philosophy:** SQL (Structured Query Language) is the language of relational databases. It is declarative — you describe WHAT you want, the database engine decides HOW to retrieve it. Understanding SQL deeply means understanding the relational model, set theory, and how queries are executed.

---

## SECTION A — SQL Fundamentals

---

### Q193. What is SQL? What are different SQL dialects?

**SQL** (Structured Query Language) is the standard language for managing and querying **relational databases**. It allows you to define structures (tables, views), manipulate data (insert, update, delete), and retrieve data (queries).

SQL is **declarative** — you describe the result you want, not the algorithm to get it.

**ANSI SQL** is the standard defined by the American National Standards Institute. Different DBMS vendors implement SQL with extensions and variations:

| Dialect | DBMS | Notable Features |
|---|---|---|
| **T-SQL** | Microsoft SQL Server, Azure SQL | `TOP`, `IDENTITY`, `GETDATE()`, CTEs before ANSI |
| **PL/pgSQL** | PostgreSQL | Procedural language, `ILIKE`, rich function set, JSONB |
| **PL/SQL** | Oracle Database | `ROWNUM`, `DUAL`, extensive procedural extensions |
| **MySQL SQL** | MySQL, MariaDB | `LIMIT`, `AUTO_INCREMENT`, `GROUP_CONCAT` |
| **SQLite** | SQLite | Minimal, file-based, serverless, `WITHOUT ROWID` |
| **SparkSQL** | Apache Spark | Distributed queries, `LATERAL VIEW` |

---

### Q194. Explain the types of SQL commands: DDL, DML, DCL, TCL.

SQL commands are classified by their purpose:

#### DDL — Data Definition Language

Defines and modifies the **structure** of database objects. Changes are **auto-committed**.

```sql
CREATE TABLE patients (
    patient_id   INT          PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    age          INT          CHECK(age BETWEEN 0 AND 150),
    glucose      DECIMAL(6,2)
);

ALTER TABLE patients ADD COLUMN bmi DECIMAL(5,2);
ALTER TABLE patients MODIFY COLUMN name VARCHAR(200);
ALTER TABLE patients DROP COLUMN bmi;

DROP TABLE patients;           -- permanently removes table
TRUNCATE TABLE patients;       -- removes all rows, keeps structure

CREATE INDEX idx_glucose ON patients(glucose);
CREATE VIEW diabetic_patients AS
    SELECT * FROM patients WHERE glucose > 140;
```

**DDL commands:** `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`, `COMMENT`

#### DML — Data Manipulation Language

Manipulates the **data** within structures. Part of a **transaction** — can be rolled back.

```sql
INSERT INTO patients (patient_id, name, age, glucose)
VALUES (1, 'Alice', 45, 165.5);

INSERT INTO patients SELECT * FROM patients_backup;  -- bulk insert

UPDATE patients SET glucose = 170.0 WHERE patient_id = 1;

DELETE FROM patients WHERE age > 90;
```

**DML commands:** `INSERT`, `UPDATE`, `DELETE`, `MERGE`

#### DCL — Data Control Language

Controls **access permissions**.

```sql
GRANT SELECT, INSERT ON patients TO 'doctor_user';
GRANT ALL PRIVILEGES ON hospital_db.* TO 'admin'@'localhost';
REVOKE INSERT ON patients FROM 'doctor_user';
```

**DCL commands:** `GRANT`, `REVOKE`

#### TCL — Transaction Control Language

Manages **transactions** — groups of DML statements that succeed or fail together.

```sql
BEGIN TRANSACTION;
    UPDATE accounts SET balance = balance - 5000 WHERE id = 101;
    UPDATE accounts SET balance = balance + 5000 WHERE id = 202;
COMMIT;    -- save both changes permanently

-- OR:
ROLLBACK;  -- undo both changes if something went wrong

SAVEPOINT sp1;
    UPDATE ...;
ROLLBACK TO SAVEPOINT sp1;   -- undo back to savepoint, not full rollback
RELEASE SAVEPOINT sp1;
```

**TCL commands:** `COMMIT`, `ROLLBACK`, `SAVEPOINT`, `RELEASE SAVEPOINT`, `SET TRANSACTION`

---

### Q195. What is a relational database? Explain the relational model.

A **relational database** stores data in **relations (tables)** — 2D structures consisting of rows (tuples) and columns (attributes). It is based on Codd's Relational Model (1970).

#### Core Concepts

```
RELATION (TABLE): patients

patient_id  name     age  glucose  is_diabetic
──────────────────────────────────────────────
1           Alice    45   165.5    TRUE
2           Bob      32   110.2    FALSE
3           Priya    58   195.0    TRUE

SCHEMA (structure): patients(patient_id, name, age, glucose, is_diabetic)
TUPLE (row):        (1, 'Alice', 45, 165.5, TRUE)
ATTRIBUTE (column): glucose
DOMAIN:             glucose → decimal values in range [0, 999.99]
DEGREE:             5 (number of columns)
CARDINALITY:        3 (number of rows)
```

#### Key Properties of Relations

1. **Each row is unique** — identified by primary key
2. **Column values are atomic** — one value per cell (1NF)
3. **Column names are unique** within a table
4. **Row order is insignificant** — a relation is a SET, not a sequence
5. **Column order is insignificant** — identified by name, not position

#### Relational Integrity

```
ENTITY INTEGRITY:    Primary key cannot be NULL
                     (every row must be uniquely identifiable)

REFERENTIAL INTEGRITY: Foreign key values must match an existing primary key
                       in the referenced table, or be NULL
                       (no orphan records)

DOMAIN INTEGRITY:    Column values must be of the correct type and range
                     (CHECK constraints, data types)
```

---

### Q196. What is a primary key? What properties must it have?

A **primary key** uniquely identifies each row in a table. It is the fundamental mechanism for row identity in the relational model.

**Required properties:**

```
1. UNIQUENESS:    No two rows can have the same primary key value.
2. NON-NULL:      Primary key columns cannot contain NULL.
3. MINIMAL:       Should use the minimum number of columns necessary (avoid bloat).
4. STABLE:        Should not change after insertion (business keys that change are bad PKs).
5. IRREDUCIBLE:   No proper subset of a composite key should also be unique.
```

**Types of primary keys:**

```sql
-- Single column (simple):
CREATE TABLE patients (
    patient_id INT PRIMARY KEY,   -- surrogate key (meaningless, system-generated)
    ...
);

-- Composite (multiple columns):
CREATE TABLE enrollment (
    student_id  INT,
    course_id   INT,
    semester    VARCHAR(10),
    PRIMARY KEY (student_id, course_id, semester)
    -- no single column uniquely identifies a row
);

-- AUTO_INCREMENT (MySQL) / IDENTITY (SQL Server) / SERIAL (PostgreSQL):
CREATE TABLE patients (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,  -- auto-generated
    name       VARCHAR(100)
);
```

**Natural key vs Surrogate key:**

| Property | Natural Key | Surrogate Key |
|---|---|---|
| Meaning | Real-world attribute (SSN, email) | System-generated ID (1, 2, 3...) |
| Stability | May change (email changes) | Never changes |
| Readability | Human-meaningful | Meaningless integer |
| Performance | Slower joins (longer key) | Faster joins (integer) |
| Recommendation | Use when stable and truly unique | Generally preferred for PKs |

---

### Q197. Explain foreign keys and referential integrity.

A **foreign key** is a column (or set of columns) in one table that references the primary key of another table, establishing a **relationship** between the two tables.

```sql
CREATE TABLE departments (
    dept_id   INT         PRIMARY KEY,
    dept_name VARCHAR(50) NOT NULL
);

CREATE TABLE employees (
    emp_id    INT         PRIMARY KEY,
    name      VARCHAR(100),
    dept_id   INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        ON DELETE SET NULL      -- if department deleted, set emp.dept_id to NULL
        ON UPDATE CASCADE       -- if dept_id changed, update in employees too
);
```

**Referential Integrity** ensures that a foreign key value always matches an existing primary key — no orphan records.

```
departments:             employees:
dept_id  dept_name       emp_id  name     dept_id
───────────────          ──────────────────────────
10       Cardiology      1       Alice    10    ✅ valid (dept 10 exists)
20       Neurology       2       Bob      20    ✅ valid
30       Oncology        3       Charlie  40    ❌ INVALID — no dept 40!
                                                   FOREIGN KEY CONSTRAINT violation
```

**ON DELETE and ON UPDATE options:**

| Option | Effect |
|---|---|
| `CASCADE` | Automatically delete/update child rows |
| `SET NULL` | Set FK to NULL in child rows |
| `SET DEFAULT` | Set FK to default value in child rows |
| `RESTRICT` | Prevent deletion/update if child rows exist |
| `NO ACTION` | Similar to RESTRICT (default in most DBMS) |

---

### Q198. What is database normalization? Why is it important?

**Normalization** is the process of organising a relational database to **reduce data redundancy** and **improve data integrity** by applying a series of formal rules (Normal Forms).

**Problems solved by normalization:**

```
UN-NORMALISED TABLE: patient_treatments

patient_id  patient_name  city      doctor_id  doctor_name   treatment  cost
───────────────────────────────────────────────────────────────────────────────
1           Alice         Mumbai    D01        Dr. Sharma    Insulin    500
1           Alice         Mumbai    D01        Dr. Sharma    Metformin  200
1           Alice         Mumbai    D02        Dr. Patel     X-Ray      1500
2           Bob           Delhi     D01        Dr. Sharma    Insulin    500

PROBLEMS:
  Update anomaly:  If Alice moves to Pune, must update MULTIPLE rows.
                   If one row missed → inconsistent data.
  
  Insert anomaly:  Cannot add a new doctor without a patient treatment.
  
  Delete anomaly:  Deleting Alice's last treatment deletes Alice's info too.
  
  Redundancy:      "Alice", "Mumbai", "Dr. Sharma" stored multiple times.
                   Wastes storage, creates inconsistency risk.
```

**After normalisation** these anomalies are eliminated by decomposing into separate tables.

---

## SECTION B — Normalization

---

### Q199. Explain 1NF, 2NF, 3NF, and BCNF with examples.

Each Normal Form builds on the previous, eliminating a specific type of dependency problem.

---

#### 1NF — First Normal Form

**Rule:** Each column must contain **atomic (indivisible) values**. No repeating groups. Each row must be uniquely identifiable.

```
VIOLATES 1NF:
patient_id  name    diagnoses
1           Alice   Diabetes, Hypertension, Obesity   ← multi-valued cell!
2           Bob     Asthma

SATISFIES 1NF:
patient_id  name    diagnosis
1           Alice   Diabetes
1           Alice   Hypertension
1           Alice   Obesity
2           Bob     Asthma
              ↑
          PK: (patient_id, diagnosis)
```

---

#### 2NF — Second Normal Form

**Rule:** Must be in 1NF AND no **partial dependency** — non-key attributes must depend on the **entire** primary key, not just part of it.

(Only relevant when PK is composite)

```
IN 1NF (composite PK: student_id + course_id):
student_id  course_id  grade  student_name  course_name
───────────────────────────────────────────────────────
1           C01        A      Alice         Math
1           C02        B      Alice         Physics
2           C01        B      Bob           Math

PARTIAL DEPENDENCIES (violate 2NF):
  student_name → depends only on student_id (not full PK)
  course_name  → depends only on course_id  (not full PK)

FIX — decompose into 3 tables:
students: (student_id, student_name)
courses:  (course_id, course_name)
enrollment:(student_id, course_id, grade)   ← only full PK dependencies
```

---

#### 3NF — Third Normal Form

**Rule:** Must be in 2NF AND no **transitive dependency** — non-key attributes must depend ONLY on the primary key, not on other non-key attributes.

```
IN 2NF:
employee_id  dept_id  dept_name    salary
────────────────────────────────────────
1            D01      Engineering  80000
2            D01      Engineering  75000
3            D02      Marketing    70000

TRANSITIVE DEPENDENCY (violates 3NF):
  employee_id → dept_id → dept_name
  dept_name depends on dept_id (non-key), not directly on employee_id

FIX:
employees:   (employee_id, dept_id, salary)
departments: (dept_id, dept_name)
```

---

#### BCNF — Boyce-Codd Normal Form

**Rule:** Must be in 3NF AND for every functional dependency X → Y, X must be a **superkey** (candidate key or superset of candidate key). Stricter than 3NF — handles anomalies 3NF misses with overlapping candidate keys.

```
VIOLATES BCNF (even though in 3NF):
student  subject    teacher
─────────────────────────────
Alice    Math       Dr. Roy
Bob      Math       Dr. Roy
Alice    Physics    Dr. Sen
Bob      Chemistry  Dr. Gupta

Candidate keys: (student, subject), (student, teacher)
Functional dependency: teacher → subject   ← teacher is NOT a superkey!
                       (each teacher teaches only one subject)

FIX:
teacher_subject: (teacher, subject)
student_teacher: (student, teacher)
```

---

### Q200. What are functional dependencies?

A **functional dependency** X → Y means that the value of attribute X **uniquely determines** the value of attribute Y. If two rows have the same X value, they must have the same Y value.

```
In the patients table:
patient_id → name           (patient_id determines name)
patient_id → age            (patient_id determines age)
patient_id → glucose        (patient_id determines glucose)
(patient_id → {name, age, glucose})  — patient_id is a determinant

But:
name       → patient_id     ← NOT necessarily (two patients can have same name)

Types:
  FULL dependency:     (student_id, course_id) → grade   (depends on BOTH)
  PARTIAL dependency:  (student_id, course_id) → student_name  (only on student_id)
  TRANSITIVE:          patient_id → zip_code → city     (city via zip, not directly)
  
Trivial FD: X → Y where Y ⊆ X  (e.g., {A,B} → A is always true, trivially)
Non-trivial FD: X → Y where Y ⊄ X  (the useful kind)
```

---

## SECTION C — Queries (Basic)

---

### Q201. Explain the structure and execution order of a SELECT statement.

**Written order vs Execution order** is one of the most important SQL concepts.

**Written (syntactic) order:**

```sql
SELECT   [DISTINCT] columns          -- 6
FROM     table_name                  -- 1
JOIN     other_table ON condition    -- 2
WHERE    row_filter_condition        -- 3
GROUP BY grouping_columns            -- 4
HAVING   group_filter_condition      -- 5
ORDER BY sort_columns                -- 7
LIMIT    n                           -- 8
```

**Execution (logical processing) order:**

```
1. FROM & JOIN   → Identify source data, perform joins
2. WHERE         → Filter individual rows
3. GROUP BY      → Group filtered rows
4. Aggregate     → Calculate COUNT, SUM, AVG etc.
5. HAVING        → Filter groups
6. SELECT        → Choose columns, apply expressions, DISTINCT
7. ORDER BY      → Sort final result
8. LIMIT/OFFSET  → Limit rows returned

WHY THIS MATTERS:
  You CANNOT use a SELECT alias in WHERE because WHERE is processed BEFORE SELECT:
    SELECT glucose * 2 AS double_glucose
    FROM patients
    WHERE double_glucose > 200;      -- ERROR! alias not yet defined at WHERE stage

  FIX: Use the expression directly or use a subquery/CTE:
    WHERE glucose * 2 > 200;         -- OK
    
  You CAN use SELECT alias in ORDER BY (ORDER BY is after SELECT):
    ORDER BY double_glucose DESC;     -- OK
```

---

### Q202. What is the difference between WHERE and HAVING clauses?

Both filter rows, but they operate at different stages of query processing.

| Property | WHERE | HAVING |
|---|---|---|
| **When applied** | BEFORE grouping (step 2) | AFTER grouping (step 5) |
| **Filters** | Individual rows | Groups (of rows) |
| **Use aggregate functions** | ❌ No (aggregation not done yet) | ✅ Yes |
| **Can use** | Column values, expressions | Aggregate results |

```sql
-- WHERE filters individual patient rows BEFORE grouping:
SELECT city, AVG(glucose) AS avg_glucose, COUNT(*) AS patient_count
FROM patients
WHERE age > 40                        -- only consider patients over 40
GROUP BY city
HAVING AVG(glucose) > 150             -- only show cities with high avg glucose
   AND COUNT(*) >= 10;                -- only cities with at least 10 patients

-- Step by step:
-- 1. FROM patients → all rows
-- 2. WHERE age > 40 → keep only patients over 40
-- 3. GROUP BY city → group those patients by city
-- 4. Aggregate: AVG(glucose), COUNT(*) per city
-- 5. HAVING → keep only groups meeting the condition
-- 6. SELECT → output city, avg_glucose, patient_count
```

**Common mistake:**

```sql
-- WRONG: Using aggregate in WHERE:
SELECT dept, SUM(salary)
FROM employees
WHERE SUM(salary) > 100000    -- ERROR! Cannot use aggregate in WHERE
GROUP BY dept;

-- CORRECT: Use HAVING:
SELECT dept, SUM(salary)
FROM employees
GROUP BY dept
HAVING SUM(salary) > 100000;  -- OK — HAVING is after aggregation
```

---

### Q203. Explain aggregate functions: COUNT, SUM, AVG, MIN, MAX.

Aggregate functions **compute a single result from multiple rows**.

```sql
SELECT
    COUNT(*)                    AS total_patients,
    COUNT(glucose)              AS patients_with_glucose,   -- excludes NULLs
    COUNT(DISTINCT city)        AS unique_cities,
    SUM(glucose)                AS total_glucose,
    AVG(glucose)                AS avg_glucose,
    MIN(glucose)                AS min_glucose,
    MAX(glucose)                AS max_glucose,
    MAX(glucose) - MIN(glucose) AS glucose_range,
    STDDEV(glucose)             AS std_deviation,           -- standard deviation
    VARIANCE(glucose)           AS variance
FROM patients
WHERE is_diabetic = TRUE;
```

**Important NULL behaviour:**
- `COUNT(*)` counts ALL rows including NULLs
- `COUNT(column)` counts only NON-NULL values in that column
- `SUM`, `AVG`, `MIN`, `MAX` all **ignore NULL values**

```sql
-- Table: scores = [100, NULL, 80, NULL, 60]

SELECT
    COUNT(*)      → 5      -- counts rows
    COUNT(score)  → 3      -- counts non-NULLs only
    SUM(score)    → 240    -- ignores NULLs: 100+80+60
    AVG(score)    → 80.0   -- 240/3 (3 non-NULLs, not 5 rows!)
    MIN(score)    → 60
    MAX(score)    → 100
```

---

### Q204. What is the GROUP BY clause? How does it work?

`GROUP BY` collapses multiple rows with the same value in the specified columns into **one row per group**, allowing aggregate functions to be applied to each group.

```sql
-- How many patients per city, and their average glucose:
SELECT
    city,
    COUNT(*)       AS patient_count,
    AVG(glucose)   AS avg_glucose,
    MAX(glucose)   AS max_glucose
FROM patients
GROUP BY city
ORDER BY avg_glucose DESC;
```

**Execution:**

```
FROM patients:
  1  Alice   Mumbai  165
  2  Bob     Delhi   110
  3  Priya   Mumbai  195
  4  Arjun   Mumbai  140
  5  Maya    Delhi   125
  6  Kiran   Chennai 180

GROUP BY city:
  Mumbai:  [Alice:165, Priya:195, Arjun:140]
  Delhi:   [Bob:110, Maya:125]
  Chennai: [Kiran:180]

AGGREGATE:
  Mumbai:  COUNT=3, AVG=166.7, MAX=195
  Delhi:   COUNT=2, AVG=117.5, MAX=125
  Chennai: COUNT=1, AVG=180.0, MAX=180

OUTPUT:
  city      count  avg_glucose  max_glucose
  Mumbai    3      166.7        195
  Delhi     2      117.5        125
  Chennai   1      180.0        180
```

**Rule:** Any column in `SELECT` that is NOT inside an aggregate function MUST appear in `GROUP BY`.

```sql
-- WRONG:
SELECT city, name, AVG(glucose) FROM patients GROUP BY city;
-- name is not in GROUP BY and not aggregated → ambiguous (which name for the group?)

-- CORRECT:
SELECT city, AVG(glucose) FROM patients GROUP BY city;
SELECT city, name, AVG(glucose) FROM patients GROUP BY city, name;
```

---

### Q205. How do NULL values behave in SQL?

NULL represents **unknown** or **missing** information — it is NOT the same as zero, empty string, or false. NULL has special behaviour that trips up many developers.

#### Comparisons with NULL

```sql
-- NULL comparisons always return UNKNOWN (not TRUE or FALSE):
NULL = NULL   → UNKNOWN (not TRUE!)
NULL != NULL  → UNKNOWN
NULL > 5      → UNKNOWN
NULL + 5      → NULL

-- CORRECT way to check for NULL:
WHERE glucose IS NULL          -- ✅
WHERE glucose IS NOT NULL      -- ✅
WHERE glucose = NULL           -- ❌ always returns UNKNOWN (finds nothing!)
```

#### NULL in WHERE Clauses

```sql
-- This finds patients where glucose is NOT 150:
SELECT * FROM patients WHERE glucose != 150;

-- But this EXCLUDES patients where glucose IS NULL!
-- NULL != 150 → UNKNOWN → row NOT returned
-- To include NULLs:
SELECT * FROM patients WHERE glucose != 150 OR glucose IS NULL;
```

#### NULL in Aggregate Functions

```sql
-- SUM, AVG, MIN, MAX, COUNT(column) all IGNORE NULLs:
SELECT AVG(glucose) FROM patients;
-- If 3 patients have glucose values (100, 200, NULL),
-- AVG = (100 + 200) / 2 = 150  (NULL ignored, denominator is 2 not 3)

-- COUNT(*) counts all rows; COUNT(glucose) skips NULLs:
SELECT COUNT(*), COUNT(glucose) FROM patients;
-- 3, 2  (one NULL glucose)
```

#### NULL in JOINs

```sql
-- INNER JOIN excludes rows where join key is NULL:
-- LEFT JOIN includes NULLs from left table (matched with NULL from right)
```

#### NULL with Functions

```sql
-- COALESCE: return first non-NULL value:
SELECT COALESCE(glucose, 0) FROM patients;     -- replace NULL with 0
SELECT COALESCE(phone, email, 'No contact') FROM contacts;  -- first non-NULL

-- NULLIF: return NULL if two values are equal:
SELECT NULLIF(denominator, 0);   -- prevent division by zero: returns NULL if 0

-- IFNULL (MySQL), NVL (Oracle), ISNULL (SQL Server):
SELECT IFNULL(glucose, 0) FROM patients;   -- MySQL equivalent of COALESCE
```

---

## SECTION D — Queries (Advanced)

---

### Q206. Explain different types of JOINs: INNER, LEFT, RIGHT, FULL OUTER.

JOINs combine rows from two or more tables based on a related column.

```
Setup tables:
patients:                   diagnoses:
id   name      dept_id      id   diagnosis   patient_id
───────────────────────      ─────────────────────────────
1    Alice      10           A    Diabetes    1
2    Bob        20           B    Hypertension 1
3    Charlie    NULL         C    Asthma      2
4    Diana      30           D    Obesity     99  ← no matching patient
```

#### INNER JOIN — Only Matching Rows

Returns rows where the join condition is satisfied in BOTH tables.

```sql
SELECT p.name, d.diagnosis
FROM patients p
INNER JOIN diagnoses d ON p.id = d.patient_id;

Result:
  Alice    Diabetes
  Alice    Hypertension
  Bob      Asthma
  -- Charlie excluded (no diagnoses)
  -- Diana excluded (no diagnoses)
  -- diagnosis D (patient 99) excluded (no matching patient)
```

#### LEFT (OUTER) JOIN — All Left + Matching Right

Returns ALL rows from the left table. For right table, returns NULL where no match.

```sql
SELECT p.name, d.diagnosis
FROM patients p
LEFT JOIN diagnoses d ON p.id = d.patient_id;

Result:
  Alice    Diabetes
  Alice    Hypertension
  Bob      Asthma
  Charlie  NULL         ← Charlie included with NULL (no diagnoses)
  Diana    NULL         ← Diana included with NULL
```

#### RIGHT (OUTER) JOIN — Matching Left + All Right

Returns ALL rows from the right table. For left table, returns NULL where no match.

```sql
SELECT p.name, d.diagnosis
FROM patients p
RIGHT JOIN diagnoses d ON p.id = d.patient_id;

Result:
  Alice    Diabetes
  Alice    Hypertension
  Bob      Asthma
  NULL     Obesity      ← diagnosis for patient 99 included, no matching patient
```

#### FULL OUTER JOIN — All Rows from Both

Returns ALL rows from BOTH tables. NULL where no match on either side.

```sql
SELECT p.name, d.diagnosis
FROM patients p
FULL OUTER JOIN diagnoses d ON p.id = d.patient_id;

Result:
  Alice    Diabetes
  Alice    Hypertension
  Bob      Asthma
  Charlie  NULL
  Diana    NULL
  NULL     Obesity
```

**CROSS JOIN — Cartesian Product**

Every row from table A paired with every row from table B. A×B = n×m rows. Rarely used (accidentally creates if WHERE join condition forgotten).

```sql
SELECT * FROM patients CROSS JOIN diagnoses;  -- 4 × 4 = 16 rows
```

---

### Q207. What is a SELF JOIN? When would you use it?

A **SELF JOIN** joins a table to itself — treating it as two separate instances. Used when relationships exist between rows within the same table.

```sql
-- Employee hierarchy: employees table has 'manager_id' column
-- referencing another employee_id in the SAME table:

employees:
emp_id  name        manager_id
───────────────────────────────
1       CEO         NULL
2       VP_Sales    1
3       VP_Tech     1
4       Sales_Rep   2
5       Developer   3

-- Find each employee and their manager's name:
SELECT
    e.name         AS employee,
    m.name         AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

Result:
  employee    manager
  CEO         NULL        (no manager)
  VP_Sales    CEO
  VP_Tech     CEO
  Sales_Rep   VP_Sales
  Developer   VP_Tech
```

**Other self join use cases:**
- Finding duplicate records
- Comparing rows within the same table (e.g., orders from same customer within 7 days)
- Hierarchical data (org charts, category trees, bill of materials)

---

### Q208. What is the difference between UNION and UNION ALL?

Both combine the result sets of two or more SELECT statements vertically (stacking rows).

| Property | UNION | UNION ALL |
|---|---|---|
| **Duplicates** | Removed (deduplication) | Kept (all rows) |
| **Performance** | Slower (needs sort/hash for dedup) | Faster (no deduplication) |
| **When to use** | Need unique results | Know results are distinct, or need duplicates |

```sql
-- UNION: removes duplicates
SELECT name FROM patients_2023
UNION
SELECT name FROM patients_2024;
-- If "Alice" appears in both, she appears ONCE in result

-- UNION ALL: keeps all rows including duplicates
SELECT name FROM patients_2023
UNION ALL
SELECT name FROM patients_2024;
-- If "Alice" appears in both, she appears TWICE in result

-- Rule: Both queries must have SAME NUMBER of columns with COMPATIBLE types
SELECT id, name, glucose FROM patients_india
UNION ALL
SELECT id, name, glucose FROM patients_usa;
```

---

### Q209. Explain subqueries: correlated vs non-correlated.

A **subquery** (inner query / nested query) is a SELECT statement embedded within another SQL statement.

#### Non-Correlated Subquery

The inner query is **independent** — it runs once, returns a result, and the outer query uses it.

```sql
-- Find patients with glucose above the average:
SELECT name, glucose
FROM patients
WHERE glucose > (SELECT AVG(glucose) FROM patients);
-- Inner query runs ONCE: AVG(glucose) = 143.5
-- Outer query: WHERE glucose > 143.5

-- Find patients in cities that have a hospital:
SELECT * FROM patients
WHERE city IN (SELECT DISTINCT city FROM hospitals);

-- Subquery in FROM (derived table):
SELECT dept, avg_sal
FROM (
    SELECT dept_id AS dept, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) AS dept_averages
WHERE avg_sal > 70000;
```

#### Correlated Subquery

The inner query **references the outer query** — it runs once for **each row** of the outer query. Slower but powerful.

```sql
-- Find patients whose glucose is above the average for THEIR CITY:
SELECT name, city, glucose
FROM patients p_outer
WHERE glucose > (
    SELECT AVG(glucose)
    FROM patients p_inner
    WHERE p_inner.city = p_outer.city    -- ← reference to outer query!
);
-- For EACH patient in the outer query, the inner query runs
-- to get AVG glucose for THAT patient's city specifically

-- Find employees who earn more than the average in their own department:
SELECT name, salary, dept_id
FROM employees e1
WHERE salary > (
    SELECT AVG(salary)
    FROM employees e2
    WHERE e2.dept_id = e1.dept_id
);
```

**Performance:** Non-correlated subqueries run ONCE. Correlated subqueries run N times (once per outer row). For large tables, prefer JOINs or window functions over correlated subqueries.

---

### Q210. What are Common Table Expressions (CTEs)?

A **CTE** is a named temporary result set defined using `WITH` that can be referenced multiple times within the same query. Think of it as a named subquery that lives for the duration of the query.

```sql
-- Without CTE (nested subqueries — hard to read):
SELECT name, glucose
FROM patients
WHERE glucose > (
    SELECT AVG(glucose)
    FROM patients
    WHERE age > 40
)
AND city IN (
    SELECT city FROM hospitals WHERE level = 'Tertiary'
);

-- With CTE (much more readable):
WITH
high_risk_threshold AS (
    SELECT AVG(glucose) AS threshold
    FROM patients
    WHERE age > 40
),
tertiary_cities AS (
    SELECT city FROM hospitals WHERE level = 'Tertiary'
)
SELECT p.name, p.glucose
FROM patients p
JOIN tertiary_cities tc ON p.city = tc.city
WHERE p.glucose > (SELECT threshold FROM high_risk_threshold);
```

**Multiple CTEs:**

```sql
WITH
city_stats AS (
    SELECT city, AVG(glucose) AS avg_g, COUNT(*) AS n
    FROM patients
    GROUP BY city
),
high_glucose_cities AS (
    SELECT city
    FROM city_stats
    WHERE avg_g > 150 AND n >= 5
)
SELECT p.*
FROM patients p
JOIN high_glucose_cities hgc ON p.city = hgc.city;
```

**Recursive CTE — for hierarchical data:**

```sql
-- Walk an employee hierarchy:
WITH RECURSIVE org_tree AS (
    -- Base case: CEO (no manager)
    SELECT emp_id, name, manager_id, 0 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees whose manager is in org_tree
    SELECT e.emp_id, e.name, e.manager_id, ot.level + 1
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.emp_id
)
SELECT level, name FROM org_tree ORDER BY level, name;
```

**CTE vs Subquery vs Temp Table:**

| Property | CTE | Subquery | Temp Table |
|---|---|---|---|
| Readability | ✅ Best | ❌ Nested = messy | Moderate |
| Reuse within query | ✅ Yes | ❌ Must repeat | ✅ Yes |
| Recursive | ✅ Yes | ❌ No | ❌ No |
| Performance | Similar to subquery | Similar to CTE | Materialised = can be faster for reuse |
| Scope | Current query only | Current query only | Session/transaction |

---

### Q211. Explain window functions (ROW_NUMBER, RANK, DENSE_RANK).

**Window functions** perform calculations across a set of rows **related to the current row** — without collapsing rows into groups (unlike `GROUP BY`). Each row keeps its identity while also having access to aggregate information about its window.

```sql
SELECT
    name,
    city,
    glucose,
    ROW_NUMBER()  OVER (PARTITION BY city ORDER BY glucose DESC) AS row_num,
    RANK()        OVER (PARTITION BY city ORDER BY glucose DESC) AS rank_num,
    DENSE_RANK()  OVER (PARTITION BY city ORDER BY glucose DESC) AS dense_rank,
    AVG(glucose)  OVER (PARTITION BY city)                       AS city_avg
FROM patients;
```

**Sample output:**

```
name     city     glucose  row_num  rank  dense_rank  city_avg
────────────────────────────────────────────────────────────────
Priya    Mumbai   195      1        1     1           166.7
Alice    Mumbai   165      2        2     2           166.7
Arjun    Mumbai   140      3        3     3           166.7
Kiran    Chennai  180      1        1     1           172.5
Meera    Chennai  165      2        2     2           172.5
Bob      Delhi    125      1        1     1           117.5
Maya     Delhi    110      2        2     2           117.5
```

**Difference between ROW_NUMBER, RANK, DENSE_RANK — with ties:**

```
Values: 100, 100, 80, 60

ROW_NUMBER:  1, 2, 3, 4  (always unique — arbitrary tie-breaking)
RANK:        1, 1, 3, 4  (ties get same rank, SKIP next number)
DENSE_RANK:  1, 1, 2, 3  (ties get same rank, NO SKIP)
```

**Other window functions:**

```sql
-- Running total:
SUM(glucose) OVER (ORDER BY patient_id) AS running_total

-- Moving average (last 3 rows):
AVG(glucose) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3

-- Lag and Lead — access previous/next row:
LAG(glucose,  1, 0) OVER (PARTITION BY patient_id ORDER BY date) AS prev_glucose
LEAD(glucose, 1, 0) OVER (PARTITION BY patient_id ORDER BY date) AS next_glucose

-- First/Last in window:
FIRST_VALUE(glucose) OVER (PARTITION BY city ORDER BY date) AS first_reading
LAST_VALUE(glucose)  OVER (PARTITION BY city ORDER BY date
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_reading

-- NTILE: divide into N buckets:
NTILE(4) OVER (ORDER BY salary) AS salary_quartile
```

**OVER clause components:**

```sql
function() OVER (
    PARTITION BY city           -- divide into groups (like GROUP BY, but rows kept)
    ORDER BY glucose DESC       -- order within each partition
    ROWS BETWEEN 2 PRECEDING    -- frame definition (what rows to include)
           AND CURRENT ROW
)
```

---

### Q212. How do you find the Nth highest value in a table?

This is one of the most frequently asked SQL interview questions.

#### Method 1 — OFFSET/LIMIT (MySQL, PostgreSQL)

```sql
-- 3rd highest glucose:
SELECT DISTINCT glucose
FROM patients
ORDER BY glucose DESC
LIMIT 1 OFFSET 2;   -- skip 2 highest, take 1
-- OFFSET N-1 for Nth highest
```

#### Method 2 — DENSE_RANK (Most Versatile)

```sql
-- Works in all major DBMS, handles ties correctly:
SELECT glucose
FROM (
    SELECT glucose,
           DENSE_RANK() OVER (ORDER BY glucose DESC) AS dr
    FROM patients
) ranked
WHERE dr = 3;   -- change 3 for different N

-- Why DENSE_RANK over RANK?
-- If top glucose values are: 195, 195, 180, 165
-- RANK:        1, 1, 3, 4  → no rank 2 exists!
-- DENSE_RANK:  1, 1, 2, 3  → 2nd highest is 180 (correct)
```

#### Method 3 — Correlated Subquery (Classic)

```sql
-- Nth highest: count N-1 values greater than current:
SELECT DISTINCT glucose
FROM patients p1
WHERE N-1 = (
    SELECT COUNT(DISTINCT glucose)
    FROM patients p2
    WHERE p2.glucose > p1.glucose
);
-- For N=3: find glucose where exactly 2 values are greater
```

#### Method 4 — CTE with Row Number

```sql
-- Per-department Nth highest salary:
WITH ranked AS (
    SELECT
        emp_id, dept_id, salary,
        DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dr
    FROM employees
)
SELECT emp_id, dept_id, salary
FROM ranked
WHERE dr = 2;   -- 2nd highest per department
```

---

## SECTION E — DDL & DML

---

### Q213. Explain CREATE, ALTER, DROP, and TRUNCATE.

```sql
-- CREATE: Define new database objects
CREATE DATABASE hospital_db;
CREATE TABLE patients (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,
    name       VARCHAR(100) NOT NULL,
    age        INT DEFAULT 0,
    glucose    DECIMAL(6,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_glucose ON patients(glucose);
CREATE VIEW diabetic_view AS
    SELECT * FROM patients WHERE glucose > 140;

-- ALTER: Modify existing structure
ALTER TABLE patients ADD COLUMN bmi DECIMAL(5,2);
ALTER TABLE patients DROP COLUMN bmi;
ALTER TABLE patients MODIFY COLUMN name VARCHAR(200) NOT NULL;
ALTER TABLE patients RENAME COLUMN glucose TO blood_glucose;
ALTER TABLE patients ADD CONSTRAINT chk_age CHECK (age BETWEEN 0 AND 150);
ALTER TABLE patients ADD CONSTRAINT fk_city
    FOREIGN KEY (city_id) REFERENCES cities(city_id);

-- DROP: Permanently remove objects
DROP TABLE IF EXISTS patients;        -- safe: IF EXISTS prevents error if not found
DROP DATABASE hospital_db;
DROP INDEX idx_glucose ON patients;
DROP VIEW diabetic_view;

-- TRUNCATE: Remove all rows, keep structure
TRUNCATE TABLE patients;
-- Faster than DELETE FROM patients (no row-by-row logging)
-- Resets AUTO_INCREMENT counter
-- Cannot be rolled back in most DBMS
-- Cannot use WHERE clause
```

---

### Q214. What is the difference between DELETE, TRUNCATE, and DROP?

This is a classic interview question requiring precise knowledge.

| Property | DELETE | TRUNCATE | DROP |
|---|---|---|---|
| **What it removes** | Selected rows (or all with no WHERE) | All rows | Entire table (structure + data) |
| **WHERE clause** | ✅ Yes | ❌ No | ❌ No |
| **Rollback** | ✅ Yes (DML — part of transaction) | ❌ No (DDL — auto-commit in most DBMS) | ❌ No |
| **Triggers fire** | ✅ Yes | ❌ No | ❌ No |
| **Speed** | Slow (row-by-row logging) | Fast (deallocates pages) | Fastest |
| **Auto-increment reset** | ❌ No | ✅ Yes | N/A |
| **Space released** | ❌ No (rows marked deleted) | ✅ Yes (extents deallocated) | ✅ Yes |
| **Command type** | DML | DDL | DDL |

```sql
-- DELETE: Remove specific rows — fully logged, rollbackable:
BEGIN TRANSACTION;
DELETE FROM patients WHERE age > 90;
-- Can ROLLBACK here
COMMIT;

-- TRUNCATE: Remove ALL rows — fast, cannot rollback (in most DBMS):
TRUNCATE TABLE temp_patients;
-- AUTO_INCREMENT resets to 1

-- DROP: Remove the table entirely — structure gone:
DROP TABLE IF EXISTS temp_patients;
-- Table no longer exists at all
```

---

### Q215. Explain constraints: PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, NOT NULL.

**Constraints** enforce rules on data in tables — they maintain data integrity automatically.

```sql
CREATE TABLE patients (
    -- PRIMARY KEY: unique identifier, NOT NULL, one per table
    patient_id    INT            PRIMARY KEY,   -- inline syntax
    -- NOT NULL: column must always have a value
    name          VARCHAR(100)   NOT NULL,
    -- UNIQUE: all values must be distinct (NULLs allowed — treated as distinct)
    email         VARCHAR(150)   UNIQUE,
    -- CHECK: enforce custom condition
    age           INT            CHECK(age >= 0 AND age <= 150),
    glucose       DECIMAL(6,2)   CHECK(glucose > 0),
    -- FOREIGN KEY: reference to another table's primary key
    city_id       INT,
    CONSTRAINT fk_city FOREIGN KEY (city_id) REFERENCES cities(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    -- Table-level PRIMARY KEY (for composite):
    -- PRIMARY KEY (patient_id, visit_date)
    -- Table-level UNIQUE:
    UNIQUE (name, age)
);
```

**Summary:**

| Constraint | Purpose | NULLs | Multiple per Table |
|---|---|---|---|
| PRIMARY KEY | Unique row identifier | ❌ Not allowed | ❌ One only |
| FOREIGN KEY | Referential integrity | ✅ Allowed (NULL = no relationship) | ✅ Many |
| UNIQUE | No duplicate values | ✅ Allowed (NULLs not compared) | ✅ Many |
| CHECK | Custom value rule | Check expression | ✅ Many |
| NOT NULL | Value required | N/A — prevents NULL | ✅ Many |

---

## SECTION F — Indexes & Performance

---

### Q216. What are indexes? Why are they important?

An **index** is a data structure (typically a B-tree or hash table) that the DBMS maintains separately from the table data to **speed up data retrieval**.

**Without index (full table scan):**

```
Query: SELECT * FROM patients WHERE email = 'alice@email.com';

Without index: DB reads every single row (10 million rows)
  → Looks at row 1: not Alice → next
  → Looks at row 2: not Alice → next
  → ...
  → Looks at row 8,347,291: FOUND Alice!
  → Continue to end (might be more Alices)
  Time: O(n) — proportional to table size
```

**With index (B-tree lookup):**

```
With index on email:
  → Binary search in B-tree index
  → Direct pointer to row(s) matching 'alice@email.com'
  Time: O(log n) — for a 10M row table, ~23 comparisons!
```

**Creating indexes:**

```sql
-- Single column:
CREATE INDEX idx_glucose ON patients(glucose);

-- Unique index (enforces uniqueness + speeds lookup):
CREATE UNIQUE INDEX idx_email ON patients(email);

-- Composite index (most selective column first):
CREATE INDEX idx_city_age ON patients(city, age);

-- Partial index (PostgreSQL):
CREATE INDEX idx_diabetic ON patients(glucose) WHERE is_diabetic = TRUE;

-- AUTO-created: primary key always gets an index; UNIQUE constraint creates unique index
```

---

### Q217. Compare clustered vs non-clustered indexes.

#### Clustered Index

**Determines the physical storage order of rows in the table.** Only ONE clustered index per table (the data can only be sorted one way).

```
Table with clustered index on patient_id:

Physical disk order:
  [patient_id=1 | name=Alice | glucose=165 | ...]
  [patient_id=2 | name=Bob   | glucose=110 | ...]
  [patient_id=3 | name=Priya | glucose=195 | ...]
  ...rows physically stored in patient_id order

B-tree index leaf nodes = actual data pages (no separate lookup)
→ Finding patient_id=500 goes directly to the row
→ Range queries (patient_id BETWEEN 100 AND 200) are very fast
   because data is physically contiguous
```

#### Non-Clustered Index

**Separate structure pointing to the actual data.** Multiple non-clustered indexes per table allowed.

```
Table physical order: (arbitrary or clustered by patient_id)
  Row 1: patient_id=1, Alice, ...
  Row 2: patient_id=2, Bob, ...
  ...

Non-clustered index on glucose (separate B-tree):
  Leaf nodes: [glucose_value → pointer to row(s) with that glucose value]
  glucose=95  → → → Row 7 (Arjun)
  glucose=110 → → → Row 2 (Bob)
  glucose=140 → → → Row 4 (Maya)
  glucose=165 → → → Row 1 (Alice)
  glucose=195 → → → Row 3 (Priya)

Finding glucose > 140: B-tree traversal → get pointers → follow to actual rows
One extra hop: index → row pointer → actual row (potential "double lookup")
```

| Property | Clustered | Non-Clustered |
|---|---|---|
| Per table | ONE only | Many (recommended max 5-10) |
| Storage | Data IS the index (leaf = row) | Separate from data |
| Range queries | Very fast (data is sorted) | Good (index sorted, data scattered) |
| Point lookup | Fast | Slightly slower (extra pointer follow) |
| INSERT/UPDATE cost | Expensive (data must stay sorted) | Cheaper |
| Default | Primary key (usually) | Explicit CREATE INDEX |

---

### Q218. When should you create an index? When should you avoid it?

#### CREATE an index when:

```sql
-- 1. Primary key (auto-created — always)
-- 2. Foreign key columns — joins become much faster:
CREATE INDEX idx_patient_city ON patients(city_id);

-- 3. Columns frequently used in WHERE:
CREATE INDEX idx_glucose ON patients(glucose);
-- Useful: SELECT * FROM patients WHERE glucose > 140;

-- 4. Columns used in JOIN conditions:
CREATE INDEX idx_emp_dept ON employees(dept_id);

-- 5. Columns used in ORDER BY (avoid sort operation):
CREATE INDEX idx_created_at ON orders(created_at DESC);

-- 6. Columns used in HAVING / GROUP BY frequently

-- 7. Composite index for common multi-column filters:
CREATE INDEX idx_city_glucose ON patients(city, glucose);
-- Helps: WHERE city = 'Mumbai' AND glucose > 140
-- The leftmost column (city) must be in the WHERE for the index to be used
```

#### AVOID an index when:

```
1. Small tables (< few thousand rows):
   Full table scan is fast enough; index overhead not worth it.

2. Columns with very low cardinality (few distinct values):
   e.g., is_diabetic (only TRUE/FALSE) — index not selective enough.
   DB might ignore the index anyway.

3. Columns rarely used in WHERE/JOIN:
   Index maintained on every INSERT/UPDATE/DELETE for no query benefit.

4. Tables with very frequent INSERT/UPDATE/DELETE:
   Every DML operation must also update all indexes → significant write overhead.
   E.g., logging tables, queue tables.

5. Columns with many NULLs (in most DBMS, NULLs not indexed):
   Index won't help queries on columns that are mostly NULL.

6. Too many indexes on same table:
   Each additional index slows down writes proportionally.
   Rule of thumb: if you have > 5-7 indexes on a write-heavy table, reconsider.
```

---

## SECTION G — Transactions

---

### Q219. What is a transaction? Explain ACID properties.

A **transaction** is a sequence of one or more SQL operations treated as a **single logical unit of work** — either ALL succeed (committed) or ALL fail (rolled back). No partial results.

**Classic example — bank transfer:**

```sql
BEGIN TRANSACTION;
    UPDATE accounts SET balance = balance - 5000 WHERE account_id = 101;
    UPDATE accounts SET balance = balance + 5000 WHERE account_id = 202;
COMMIT;   -- both succeed: permanent

-- If server crashes between the two updates:
ROLLBACK; -- undo the first update — account 101 gets its money back
```

#### ACID Properties

**A — Atomicity**

The transaction is all-or-nothing. Either ALL operations succeed and are committed, or if ANY operation fails, ALL are rolled back. No partial state.

```
Transfer of ₹5000 from A to B:
  Debit A: ✅
  Credit B: ❌ (network error)
  → ROLLBACK: Debit A is undone. A still has full balance.
  → No ₹5000 disappears into the void.
```

**C — Consistency**

A transaction brings the database from one VALID state to another VALID state. All data integrity rules (constraints, triggers, cascades) must be satisfied before and after the transaction.

```
CONSISTENCY example:
  Rule: total money in bank must always be constant
  Before: Account A = 10000, Account B = 5000 → total = 15000
  Transfer ₹3000: A = 7000, B = 8000 → total = 15000 ✅
  If transfer violated a CHECK constraint → transaction rolled back
```

**I — Isolation**

Concurrent transactions execute as if they were serial — each transaction cannot see intermediate states of other uncommitted transactions. The degree of isolation is configurable (see isolation levels).

```
ISOLATION example:
  Transaction 1: reads balance = 10000, computes new balance, writes 7000
  Transaction 2: concurrently reads balance

  WITHOUT ISOLATION: T2 might read 10000 (old) or 7000 (new)
                     depending on timing — inconsistent!
  WITH ISOLATION:    T2 reads a consistent snapshot
```

**D — Durability**

Once a transaction is committed, it persists permanently — even if the system crashes immediately after. Achieved through write-ahead logging (WAL).

```
DURABILITY example:
  COMMIT;  ← server crashes 1 millisecond later
  After recovery: the committed changes are still there
  (WAL ensures committed data is written to durable storage before confirming COMMIT)
```

---

### Q220. Explain isolation levels: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE.

Isolation levels control the trade-off between **data consistency** and **concurrency performance**. Higher isolation = more consistency, less concurrency.

**Three phenomena isolation levels prevent:**

```
DIRTY READ:
  T1 writes uncommitted data.
  T2 reads T1's uncommitted data.
  T1 rolls back.
  T2 used data that never existed!

NON-REPEATABLE READ:
  T1 reads row A → gets value 100
  T2 modifies and commits row A → value becomes 150
  T1 reads row A again → gets 150
  T1 saw two different values for the same row in same transaction!

PHANTOM READ:
  T1 queries "SELECT * WHERE age > 40" → gets 5 rows
  T2 inserts new row with age=45, commits
  T1 queries "SELECT * WHERE age > 40" again → gets 6 rows!
  New "phantom" row appeared between reads.
```

#### The Four Isolation Levels

**READ UNCOMMITTED** — Lowest isolation

Can read uncommitted changes from other transactions (dirty reads).

```
T1: UPDATE patients SET glucose = 999 WHERE id = 1;
    -- NOT YET COMMITTED
T2: SELECT glucose FROM patients WHERE id = 1;
    -- Returns 999 (dirty read — T1 might roll back!)
T1: ROLLBACK;  -- oops
```
- Prevents: Nothing
- Allows: Dirty reads, Non-repeatable reads, Phantom reads
- Use: Almost never — only for approximate/analytics queries where accuracy doesn't matter

---

**READ COMMITTED** — Most common default (PostgreSQL, Oracle, SQL Server default)

Only reads committed data. Each read statement sees the latest committed snapshot at that point in time.

```
T1: UPDATE patients SET glucose = 999;
    -- NOT YET COMMITTED
T2: SELECT glucose FROM patients;
    -- Returns ORIGINAL value (committed) ✅ No dirty read

T1: COMMIT;
T2: SELECT glucose FROM patients;  -- second read in same transaction
    -- Returns 999 (T1 is now committed — non-repeatable read can occur)
```
- Prevents: Dirty reads ✅
- Allows: Non-repeatable reads, Phantom reads

---

**REPEATABLE READ** — MySQL InnoDB default

Rows read within a transaction remain consistent throughout the transaction — same row read twice returns same value even if another transaction modified and committed it.

```
T1: SELECT glucose FROM patients WHERE id = 1;  → 165
T2: UPDATE patients SET glucose = 200 WHERE id = 1; COMMIT;
T1: SELECT glucose FROM patients WHERE id = 1;  → still 165 ✅
```
- Prevents: Dirty reads ✅, Non-repeatable reads ✅
- Allows: Phantom reads (new rows can appear in range queries)

---

**SERIALIZABLE** — Highest isolation

Transactions execute as if they ran serially, one after another. No concurrency anomalies possible.

```
T1: SELECT COUNT(*) FROM patients WHERE glucose > 140; → 5
T2: INSERT INTO patients VALUES (...glucose=195...); -- blocked until T1 completes
T1: SELECT COUNT(*) FROM patients WHERE glucose > 140; → still 5 ✅ (no phantom)
T1: COMMIT;
T2: Now runs.
```
- Prevents: All anomalies ✅
- Cost: Lowest concurrency — transactions may block each other heavily, deadlocks possible
- Use: Financial calculations, audits, anything where perfect consistency is mandatory

**Summary table:**

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|---|---|---|---|---|
| READ UNCOMMITTED | ✅ Possible | ✅ Possible | ✅ Possible | Highest |
| READ COMMITTED | ❌ Prevented | ✅ Possible | ✅ Possible | High |
| REPEATABLE READ | ❌ Prevented | ❌ Prevented | ✅ Possible | Medium |
| SERIALIZABLE | ❌ Prevented | ❌ Prevented | ❌ Prevented | Lowest |

```sql
-- Setting isolation level:
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION;
    SELECT ...;
    UPDATE ...;
COMMIT;
```

---

> **The one insight that connects Java, Python, and SQL:**
> *Java gives you structure and type safety — everything is a class, everything has a type.
> Python gives you flexibility and expressiveness — everything is an object, types are optional.
> SQL gives you declarative power — describe what you want from structured data, not how to get it.*
>
> *Together, these three form the core toolkit of any modern software engineer or data scientist.
> Master the fundamentals — OOP, memory, collections in Java;
> generators, decorators, and the scientific stack in Python;
> query optimisation, joins, and transactions in SQL —
> and you can reason clearly about almost any technical problem you encounter.*