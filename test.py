
if __name__ == '__main__':
  str = "ASDqweASD"
  new_str = ""
  for i in range(len(str)):
    if ord(str[i]) >= 65 | ord(str[i]) <= 90 :
      new_str += str[i]
  print(new_str)

