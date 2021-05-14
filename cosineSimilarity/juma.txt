def palindrome(s):
    pk=int(s)+int(s[::-1])
    pk=str(pk)
    if pk==pk[::-1]:return pk
    else:return palindrome(pk)
hello=input("enter a num :")
palindrome(hello)