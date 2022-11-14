// compile the file with: gcc -o splitimage splitimage.c -ljpeg
/*
ffmpeg -i stitch_mjpeg.avi -c:v copy -f mjpeg - | splitimage -s 4 | ffmpeg -f image2pipe -r 5 -i - -c:v copy -y splitimage_mjpeg.avi
ffmpeg -i stitch_mjpeg.avi -c:v copy -f mjpeg - | splitimage -s 2 | ffmpeg -f image2pipe -r 10 -i - -c:v copy -y splitimage_mjpeg2.avi
*/

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#define MSGSIZE 400000  // 4MB buffer
#define TRUE 1
#define FALSE 0
 //   cat log.txt | ./splitimage > log2.txt
int step = 4;

// parse the args to get the step in '-s' arg
void parse_args(int argc, char *argv[])
{
    int opt = 0;
    while ((opt = getopt(argc, argv, "s:")) != -1) {
        switch (opt) {
        case 's':
            step = atoi(optarg);
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s [-s step]", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}


int main(int argc,char* argv[])
{
    // parse the args
    parse_args(argc, argv);

    char * inbuf = (char *)malloc(MSGSIZE);
    long charpointer = 0;
    char end_1 = '\xFF';
    int end_2 = '\xD9';
    size_t max_linelen = 1000;
  
    
    /* continued */
    // while not end of stdin
    int file_not_end = TRUE;
    // FILE *input = fopen("log.txt", "r");
    FILE *input = stdin;
    for (uint i=0; file_not_end; i++) {
        // while not end of stdin, loop each image
        int do_write = i%step == 0;
        while (TRUE){
            // read the full buffer of the image
            char *line = NULL;
            size_t nread = getdelim(&line, & max_linelen, end_2, input);
            if (nread == -1) { //end of file
                file_not_end = FALSE;
                break;
            }
            else {
                // if (do_write){
                    char * dest = inbuf+charpointer;
                    memcpy(dest, line, nread);
                    charpointer += nread;
                // }
                if (nread>=2 && line[nread-2] == end_1) {
                    break;
                }
            }
            free(line);
        }
        // write the full image buffer to stdout
        if (do_write) {
            write(STDOUT_FILENO, inbuf, charpointer);
        }
        charpointer = 0;
    }
    free(inbuf); 
    
    return 0;
}