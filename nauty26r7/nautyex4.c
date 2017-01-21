/* This program prints generators for the automorphism group of an
   n-vertex polygon, where n is a number supplied by the user.
   This version uses sparse form with dynamic allocation.
*/

#include "nausparse.h"    /* which includes nauty.h */

void print_adjlist(sparsegraph sg)
{
  for (int i = 0; i < sg.vlen; i++)
  {
    printf("%d: ", i);
    for (int j = sg.v[i]; j < sg.v[i] + sg.d[i]; j++)
    {
      printf("%d, ", sg.e[j]);
    }
    printf("\n");
  }
}

int
main(int argc, char *argv[])
{
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_SPARSEGRAPH(options);
    statsblk stats;
    sparsegraph sg;   /* Declare sparse graph structure */
    sparsegraph canong;

    int n,m,i;

    options.writeautoms = TRUE;
    options.getcanon = 1;

 /* Initialise sparse graph structure. */

    SG_INIT(sg);
    SG_INIT(canong);

    while (1)
    {
        printf("\nenter n : ");
        if (scanf("%d",&n) == 1 && n > 0)
        {
            m = SETWORDSNEEDED(n);
            nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

            DYNALLOC1(int,lab,lab_sz,n,"malloc");
            DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
            DYNALLOC1(int,orbits,orbits_sz,n,"malloc");

         /* SG_ALLOC makes sure that the v,d,e fields of a sparse graph
            structure point to arrays that are large enough.  This only
            works if the structure has been initialised. */

            SG_ALLOC(sg,n,2*n,"malloc");
            SG_ALLOC(canong,n,2*n,"malloc");

            sg.nv = n;              /* Number of vertices */
            sg.nde = 2*n;           /* Number of directed edges */

            for (i = 0; i < n; ++i)
            {
                sg.v[i] = 2*i;
                sg.d[i] = 2;
                sg.e[2*i] = (i+n-1)%n;      /* edge i->i-1 */
                sg.e[2*i+1] = (i+n+1)%n;    /* edge i->i+1 */
            }

            /* an automorphic graph for 5 vertices */
            /*
            sg.e[0] = 2;
            sg.e[1] = 1;
            sg.e[2] = 0;
            sg.e[3] = 3;
            sg.e[4] = 0;
            sg.e[5] = 4;
            sg.e[6] = 1;
            sg.e[7] = 4;
            sg.e[8] = 3;
            sg.e[9] = 2;
            */

            printf("Generators for Aut(C[%d]):\n",n);
            sparsenauty(&sg,lab,ptn,orbits,&options,&stats, &canong);

            printf("Automorphism group size = ");
            writegroupsize(stdout,stats.grpsize1,stats.grpsize2);
            printf("\n");

            sortlists_sg(&canong);
            print_adjlist(sg);
            print_adjlist(canong);
        }
        else
            break;
    }

    exit(0);
}
