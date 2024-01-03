// Template for parsing an ELF file to print its symbol table
// UPDATED: Tue Dec  8 03:27:18 PM CST 2020 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <elf.h>

// The below macros help to prevent errors when doing pointer
// arithmetic. Adding an offset to a pointer is the most common
// operation here while the other macros may not be needed.

#define PTR_PLUS_BYTES(ptr,off) ((void *) (((size_t) (ptr)) + ((size_t) (off))))
// macro to add a byte offset to a pointer, arguments are a pointer
// and a # of bytes (usually size_t)

#define PTR_MINUS_BYTES(ptr,off) ((void *) (((size_t) (ptr)) - ((size_t) (off))))
// macro to subtract a byte offset from a pointer, arguments are a pointer
// and a # of bytes (usually size_t)

#define PTR_MINUS_PTR(ptr,ptq) ((long) (((size_t) (ptr)) - ((size_t) (ptq))))
// macro to subtract one pointer from another

int DEBUG = 0;                  
// Controls whether to print debug messages
// Can be used in conditionals like if(DEBUG){ ... }
// and running 'showsym -d x.o' will set DEBUG=1

int main(int argc, char *argv[]){
  if(argc < 2){
    printf("usage: %s [-d] <file>\n",argv[0]);
    return 0;
  }

  char *objfile_name = argv[1];

  // check for debug mode
  if(argc >=3){
    if(strcmp("-d",argv[1])==0){
      DEBUG = 1;
      objfile_name = argv[2];
    }
    else{
      printf("incorrect usage\n");
      return 1;
    }
  }
                        
  // Open file descriptor and set up memory map for objfile_name
  int fd = open(objfile_name, O_RDONLY);
  struct stat stat_buf;
  fstat(fd, &stat_buf);                            //get stats on the open file
  int size = stat_buf.st_size;                     //size for memory map

  // pointer to file contents; call mmap with given size and file descriptor; read only, not shared, offset 0
  char *file_bytes =  mmap(NULL, size, PROT_READ, MAP_PRIVATE,fd, 0);                                                                             

  // CREATE A POINTER to the intial bytes of the file which are an ELF64_Ehdr struct
  Elf64_Ehdr *ehdr = (Elf64_Ehdr *) file_bytes; // binary header struct is first thing in the file                                               


  // CHECK e_ident field's bytes 0 to for for the sequence {0x7f,'E','L','F'}.
  // Exit the program with code 1 if the bytes do not match
  int ident_matches =                                   // check the first bytes to ensure correct file format
    ehdr->e_ident[0] == 0x7f &&
    ehdr->e_ident[1] == 'E'  &&
    ehdr->e_ident[2] == 'L'  &&
    ehdr->e_ident[3] == 'F';

  if(!ident_matches){                                                               //check the first bytes to ensure correct file format 
    printf("ERROR: Magic bytes wrong, this is not an ELF file\n");
    exit(1);
  }


  // PROVIDED: check for a 64-bit file
  if(ehdr->e_ident[EI_CLASS] != ELFCLASS64){
    printf("Not a 64-bit file ELF file\n");
    return 1;
  }

  // PROVIDED: check for x86-64 architecture
  if(ehdr->e_machine != EM_X86_64){
    printf("Not an x86-64 file\n");
    return 1;
  }

  // DETERMINE THE OFFSET of the Section Header Array (e_shoff), the
  // number of sections (e_shnum), and the index of the Section Header
  // String table (e_shstrndx). These fields are from the ELF File
  // Header.
  int e_shoff = ehdr->e_shoff;
  int e_shnum = ehdr->e_shnum;
  int e_shstrndx = ehdr->e_shstrndx;

  // Set up a pointer to the array of section headers. Use the section
  // header string table index to find its byte position in the file
  // and set up a pointer to it.
  Elf64_Shdr *sec_hdrs = (Elf64_Shdr *) (file_bytes + e_shoff);
  char *sec_names = NULL;

  
  // Search the Section Header Array for the secion with name .symtab
  // (symbol table) and .strtab (string table).  Note their positions
  // in the file (sh_offset field).  Also note the size in bytes
  // (sh_size) and and the size of each entry (sh_entsize) for .symtab
  // so its number of entries can be computed.
  

  int pos_symtab = -1;   //this will be the index in section header where .symtab is found
  int pos_strtab = -1;   //this will be the index in section header where .strtab is found

  for(int i=0; i<e_shnum; i++){
    //printf("%ld\n", sec_hdrs[e_shstrndx].sh_offset + sec_hdrs[i].sh_name);
    sec_names = PTR_PLUS_BYTES(file_bytes, sec_hdrs[e_shstrndx].sh_offset + sec_hdrs[i].sh_name);     

    if(strcmp(sec_names, ".symtab") == 0) {     
      pos_symtab = i;                                //index in section header where .symtab is found
    }

    if(strcmp(sec_names, ".strtab") == 0) {            
      pos_strtab = i;                                 //index in section header where .strtab is found
    }    
  }

  if(pos_symtab < 0){                                      //pos_symtab negative means that symbol table was not found
    printf("ERROR: Couldn't find symbol table\n");
    return 1;
  }

  if(pos_strtab < 0){                                        //pos_strtab negative means that string table was not found
    printf("ERROR: Couldn't find string table\n");
    return 1;
  }

  
  // PRINT byte information about where the symbol table was found and
  // its sizes. The number of entries in the symbol table can be
  // determined by dividing its total size in bytes by the size of
  // each entry.

  
  printf("Symbol Table\n");
  printf("- %ld bytes offset from start of file\n", sec_hdrs[pos_symtab].sh_offset);
  printf("- %ld bytes total size\n",sec_hdrs[pos_symtab].sh_size);
  printf("- %ld bytes per entry\n",sec_hdrs[pos_symtab].sh_entsize);
    
  long int symtab_num = sec_hdrs[pos_symtab].sh_size / sec_hdrs[pos_symtab].sh_entsize;
  printf("- %ld entries\n", symtab_num);


  // Set up pointers to the Symbol Table and associated String Table
  // using offsets found earlier.
  Elf64_Sym *sym_tab = (Elf64_Sym *) (file_bytes + sec_hdrs[pos_symtab].sh_offset);
  char *str_value = NULL;

  /*possible symbol type*/
  const char * symbol_type[] = {
    "NOTYPE",
    "OBJECT",
    "FUNC",   
    "SECTION",
    "FILE"
  };

  // Print column IDs for info on each symbol
  printf("[%3s]  %8s %4s %s\n",
         "idx","TYPE","SIZE","NAME");

  // Iterate over the symbol table entries
  for(int i=0; i < symtab_num; i++){
    
    /*point to string table location in the file. 
    notice the offset will be the offset for the string table (i.e. sec_hdrs[pos_strtab].sh_offset)
    plus the offset given the symbol table (i.e. sym_tab[i].st_name)*/
    str_value = PTR_PLUS_BYTES(file_bytes, sec_hdrs[pos_strtab].sh_offset + sym_tab[i].st_name);

    // Determine size of symbol and name. Use <NONE> name has zero
    // length.
    if(strlen(str_value) == 0) {
      str_value = "<NONE>";
    }

    // Determine type of symbol. See assignment specification for
    // fields, macros, and definitions related to this.
    unsigned char typec = ELF64_ST_TYPE(sym_tab[i].st_info);

    // Print symbol information
    printf("[%3d]: %8s %4lu %s\n", 
           i,                                     //index
           symbol_type[typec],                    //type of symbol. typec 
           sym_tab[i].st_size,                    //size
           str_value);                            //name
  }  

  // Unmap file from memory and close associated file descriptor 
  munmap(file_bytes, size);                  // unmap and close file
  close(fd);

  return 0;
}