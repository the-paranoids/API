import sys

def return_tuples(array) :
    size = len(array)
    if(size%3) != 0 :
        print("Error")
        return 0

    lst = []
    stop = size / 3;
    lst.append('[')
    for i in range(0,size,3) :
        lst.append('(')
        lst.append(array[i])
        lst.append(', ')
        lst.append(array[i+1])
        lst.append(', ')
        lst.append(array[i+2])
        lst.append(')')
        if(i<=stop) :
            lst.append(', ')
    lst.append(']')
    return lst

array2 = []
for i in range(1,len(sys.argv)) :
    array2.append(str(sys.argv[i]))

lst2 = return_tuples(array2)

print("\n\nCommand Line Arguments: ")

for i in range(len(lst2)) :
    print(lst2[i],end="")
print("\n\n")

