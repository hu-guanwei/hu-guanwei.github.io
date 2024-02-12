---
layout: post
title:  "On String Replacement"
date:   2024-02-12 16:08:00 +0800
categories: interview-question
---


**Q**: given an input string, replace substring `target` with `repl`.

Notice that the result can be longer or shorter than input. (1) when result is shorter, can do it inplace, and using only one pass. (2) when longer, need to allocate extral space (1st pass), and do comparison and copy in the 2nd pass. 

{% highlight c %}
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

char* str_replace(char *input, char *target, char *repl) {
    size_t input_sz = strlen(input);
    size_t target_sz = strlen(target);
    size_t repl_sz = strlen(repl);

    // result is not longer than input
    if (repl_sz <= target_sz) {
        // loop invariant:
        //   input[0...i-1] processed
        //   input[i...j-1] free space to write
        //   input[j...n-1] to be read
        // i: slow pointer, position to write; j: fast pointer, position to read

        size_t i = 0;
        size_t j = 0;

        while (j < input_sz + 1) {
            if (!memcmp(input + j, target, target_sz)) {
                memcpy(input + i, repl, repl_sz);
                i += repl_sz;
                j += target_sz;
            } else {
                input[i++] = input[j++];
            }
        }
        return input;
    } else {
        // result is longer than input
        size_t cnt = 0;
        int k = 0;
        while (k < input_sz) {
            if (!memcmp(input + k, target, target_sz)) {
                k += target_sz;
                cnt += 1;
            } else {
                k += 1;
            }
        }

        size_t expand_sz = cnt * (repl_sz - target_sz);
        char *output = (char *)realloc(input, input_sz + expand_sz + 1);
        if (output == NULL) {
            printf("Failed realloc\n");
            return output;
        }
        output[input_sz + expand_sz] = '\0';
        int i = input_sz + expand_sz - 1;
        int j = input_sz - 1;
        // i: position to write
        // j: position to read
        // this time, go backwards

        while (j >= 0) {
            if (!memcmp(output + j - target_sz + 1, target, target_sz)) {
                memcpy(output + i - repl_sz + 1, repl, repl_sz);
                i -= repl_sz;
                j -= target_sz;
            } else {
                output[i--] = output[j--];
            }
        }
        return output + i + 1;
    }
}
{% endhighlight c %}