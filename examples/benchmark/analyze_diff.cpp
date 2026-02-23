#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <file1> <file2>\n", argv[0]);
        return -1;
    }
    
    FILE* f1 = fopen(argv[1], "rb");
    FILE* f2 = fopen(argv[2], "rb");
    
    if (!f1 || !f2) {
        printf("Failed to open files\n");
        return -1;
    }
    
    float v1, v2;
    int count = 0;
    int mismatches = 0;
    
    while (fread(&v1, sizeof(float), 1, f1) == 1 && fread(&v2, sizeof(float), 1, f2) == 1) {
        if (fabs(v1 - v2) > 1e-4) {
            if (mismatches < 20) {
                 printf("Idx %d: Base=%f, Opt=%f, Diff=%f\n", count, v1, v2, v1-v2);
            }
            mismatches++;
        }
        count++;
    }
    
    printf("Total Mismatches (>1e-4): %d / %d\n", mismatches, count);
    
    fclose(f1);
    fclose(f2);
    return 0;
}
