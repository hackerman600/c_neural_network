#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

struct neuron_layer {
    int no_neurons;
    char * activation;
};

struct weight_layer {
    int rows;
    int cols;
    double **weights;
};

struct z_out {
    int rows;
    int cols;
};

double** initialise_w_matrix(int rows, int cols){
    double** w = malloc(sizeof(double*)*rows);
    for (int r = 1; r <= rows; r ++){
        double *rr = malloc(sizeof(double)*cols);
        w[r-1] = rr;
        for (int c = 1; c <= cols; c ++){
            w[r-1][c-1] = (double)c;             
        } 
    } 
    return w;
}; 

double*** initialise_all_weights(struct weight_layer* W_d, int No_W){
    double*** full_w_matrix = malloc(sizeof(double**) * No_W);
    for (int w = 0; w < No_W; w++){
        double **W = malloc(sizeof(double*) * W_d[w].rows * W_d[w].cols + 1);
        full_w_matrix[w] = W;
        for (int rows = 0; rows < W_d[w].rows; rows++){
            double *rowz = malloc(sizeof(double) * W_d[w].cols);
            full_w_matrix[w][rows] = rowz;
            for (int cols = 0; cols < W_d[w].cols; cols++){
                if (cols != 0 && cols % 3 == 0){
                    full_w_matrix[w][rows][cols] = (rand() % 9)*.134243 * - 1;
                } else{
                    full_w_matrix[w][rows][cols] = (rand() % 9)*.134243;   
                }
                 
            }
        }
    }    
    
    return full_w_matrix;
}


double* initialise_z(int size){
    double* arr = (double*)malloc(sizeof(double) * size);
    for (int i = 1; i <= 3; i++){
        arr[i-1] = i;
    }
    return arr;
};


double** z_feed_forward(double **weights, double **data, int rows, int cols){
    double **Z = malloc(sizeof(double*)*100);

    for (int training_e = 0; training_e < 100; training_e++){
        double* r1 = malloc(sizeof(double)*cols);
        Z[training_e] = r1;
        for (int c = 0; c < cols; c++){
            double Zee = 0.0;
            for (int r = 0; r < rows; r++){
                Zee += data[training_e][r] * weights[r][c];
                //printf("%f,%s", Zee," ");
            }
            Z[training_e][c] = Zee;
            Zee = 0.0;
            //printf("\n");
        }
    }   

    return Z;

}




double* relu(double *z, int cols){
    double *outp = malloc(sizeof(double) * cols);
    for (int i = 0; i < cols; i++){
        if (z[i] < 0.0){
            outp[i] = 0.0;
        } else {
            outp[i] = (double)z[i];
        }
    }
    return outp;
}


void print_model_structure(struct neuron_layer* model_structure, struct weight_layer* weight_dimension, int size){
    
    printf("\n--------- model_structure summary ---------\n\n");
    
    int training_e = 100;
    for (int i = 1; i < size; i++){

        printf("%s,%d,%s%d%s%d%s", "Weight ",i," ( ", weight_dimension[i-1].rows, ", ", weight_dimension[i-1].cols,")\n");            
        printf("%s,%d,%s,%d,%s%s%s%d,%d%s,%s", "Layer ",i," no_neurons ", model_structure[i].no_neurons, " activation: ", model_structure[i].activation, "(",100, weight_dimension[i-1].cols,")","\n\n");
        
    }  
}


struct weight_layer* create_weight_dimension(struct neuron_layer* model_structure,int no_layer){
    struct weight_layer* weight_dimension = malloc(sizeof(struct weight_layer)*no_layer-1); 
    for (int i = 0; i < no_layer - 1; i++){
        struct weight_layer W;
        W.rows = model_structure[i].no_neurons;
         W.cols = model_structure[i + 1].no_neurons;
        weight_dimension[i] = W;
        
    }
    return weight_dimension;
};

double* initialise_bias(int no_layer){
    srand(time(NULL));
    double* b = malloc(sizeof(double)*no_layer);
    for (int i = 0; i < no_layer; i++){
        double r = rand() % 20;
        b[i] = r;
    }

    return b;
    
}; 


void print_weight_matrix(double** W, int rows, int cols){
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            printf("%f,%s", W[r][c]," ");
        }
        printf("%s","\n");
    }
    printf("%s","\n\n");
};


double*** generate_dataset(){
    srand(time(NULL));
    double*** data = malloc(sizeof(double**)*2);
    double** d = malloc(sizeof(double*) * 100);
    double** d1 = malloc(sizeof(double*) * 20);
    data[0] = d;
    data[1] = d1;

    for (int x = 0; x < 120; x++){
        double *row = malloc(sizeof(double) * 5);
        if (x < 100){
            data[0][x] = row;
        } else {
            data[1][x-100] = row;
        }       
        for (int c = 0; c < 5; c ++){
            if (x < 100){
                data[0][x][c] = rand() % 2;
            } else {
                data[1][x-100][c] = rand() % 2;
            }   
        }

    }    

    return data;
}


void print_x_data(double **x_data,int rows,int cols){
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            printf("%f,%s", x_data[r][c]," ");
        }
        printf("\n");
     }

     printf("\n\n");
}

double* generate_y(double** x_vals, int rows, int cols){
    double *y = malloc(sizeof(double) * rows);
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            if (c == 0 && x_vals[r][c] == 1){
                y[r] = 1;
            } else {
                y[r] = 0;
            }
        }
    }

    return y; 
}


double* soft_max(double *pred, int cols){
    double* soft_out = malloc(sizeof(double)*cols);
    double sum = 0.0;
    for (int i = 0; i < cols; i++){
        sum += exp(pred[i]);
    }
    for (int i = 0; i < cols; i++){
        soft_out[i] = exp(pred[i])/sum;
    }

    return soft_out;
        
};


double* onehot_encode(double **soft_y_pred, double *y, int rows, int cols){
    double *onehot_out = malloc(sizeof(double)*rows);

    for (int x = 0; x < rows; x++){
        if (soft_y_pred[x][0] > soft_y_pred[x][1]){
            onehot_out[x] = 0;
        }

        else if (soft_y_pred[x][1] > soft_y_pred[x][0]){
            onehot_out[x] = 1;
        }

        else { 
            onehot_out[x] = 1;
        }
    }

    return onehot_out;
}


double* sparse_cat_loss_vector(double **soft_y_pred, double *y, int rows, int cols){
    double *sparse_cat_crossent = malloc(sizeof(double)*rows); 
    double total_cost = 0.0;
    double *y_pred_onehot = onehot_encode(soft_y_pred, y, rows, cols);
    int tally = 0.0;
    double rw = (double)rows;
    for (int r = 0; r < rows; r++){
        int id = (int)y[r];
        double cost = -log(soft_y_pred[r][id]);

        if (y_pred_onehot[r] == y[r]){
            tally += 1.0;
        }

        if (cost < 0){
            cost *= -1;
        } if (isinf(cost)){
            cost = 2.0;
        }

        sparse_cat_crossent[r] = cost;
        total_cost += cost;
        printf("cost: %f,%s",cost,"\n");
    }
    printf("\ntotal loss is: %f", total_cost/rows);
    printf("\ntotal accuracy is: %f", tally/rw*100.0);

    return sparse_cat_crossent;
}

double *** calculate_gradients(double ***weights, double *bias, double ***a_out, double ***z_out, struct neuron_layer* model_structure, struct weight_layer* weight_dimension){
    double ***gradients = malloc(sizeof(double**)*4);
    /*calculus behind network 
    
    dZ4 = -log(p) = -1/p  ---->  (100,2)
    dW4 = A3.T (4,100) * dZ4 (100,2) ---->  (4,2)
    dB4 = 1 * sum(dZ4);
    
    dZ3 = dZ4 (100,2) * W4.T (2,4) * relu(1,0 -> Z3) ----> (100,4)
    dW3 = A2.T (5,100) * Z3 (100,4) *  ----> (5,4)
    dB3 = 1 * sum(dZ3);

    dZ2 = dZ3 (100,4) * W3.T (4, 5)  * relu(1,0 -> Z2) ----> (100,5);
    dW2 = A1.T (5,100) * dZ2 (100,5); ----> (5,5);
    dB2 = 1 * sum(dZ2);

    dZ1 = dZ2.T (100,5) * W2.T (5, 5) *  relu(1,0 -> Z1) ----> (100,5);
    dW1 = A1.T (5,100) * dZ1.T (100,5); ----> (5,5);
    dB1 = 1 * sum(dZ1);

    */
    

};



double ** activate_zlayer(double **z_outs, int weight_cols,int z_layer_index){
    double **activate_l = malloc(sizeof(double*)*100);  
    printf("z_layer_index = ,%d",z_layer_index);   

    for (int z_rows = 0; z_rows < 100; z_rows++){
        double *row = z_outs[z_rows];
        if (z_layer_index == 3){
           double *out = soft_max(row,weight_cols); 
           activate_l[z_rows] = out;
        } else {
           double *out = relu(row,weight_cols); 
           activate_l[z_rows] = out;
        }
        
    }        

    return activate_l;

}


int main(){

    //CREATE MODEL STRUCTURE;
    int no_layer = 5;
    struct neuron_layer* model_structure = malloc(sizeof(struct neuron_layer)*no_layer); 
    int num_neurons[] = {5,5,5,4,2};  
    
    for (int i = 0; i < no_layer; i++){
        struct neuron_layer Li; 
        Li.no_neurons = num_neurons[i];
        if (i == 4){
            Li.activation = "softmax";
        } else {
            Li.activation = "relu";
        }
        model_structure[i] = Li;
    }

    struct weight_layer* weight_dimension = create_weight_dimension(model_structure,no_layer);
    print_model_structure(model_structure,weight_dimension,no_layer);

    //SET THE DATASET
    //char *columns[5] = {"does it meow", "size", "tail","gender","desexed"}; 
    printf("\n\n--------------------- dataset ---------------------\n\n");
    double ***x_data = generate_dataset(); 
    double *y_data = generate_y(x_data[0],100,5);

    print_x_data(x_data[0],100,5);


    //INITIALISE THE WEIGHTS, BIAS
    printf("\n\n--------------------- weights ---------------------\n\n");
    double *bias = initialise_bias(no_layer);
    double ***weights = initialise_all_weights(weight_dimension,no_layer);

    for (int w = 0; w < 4; w++){
        print_weight_matrix(weights[w],weight_dimension[w].rows,weight_dimension[w].cols);
    }

    //TEST 1 FEED FORWARD
    double ***z_outs = malloc(sizeof(double **) * no_layer);
    double ***a_outs = malloc(sizeof(double **) * no_layer);

    printf("\n\n------------------------- z_output from each output -------------------------\n\n\n");

    for (int x = 0; x < no_layer - 1; x++){
        if (x == 0){
            double** Z = z_feed_forward(weights[x], x_data[0], weight_dimension[x].rows, weight_dimension[x].cols);
            z_outs[x] = Z;
            printf("\n\n");
        } else {
            double** Z = z_feed_forward(weights[x], z_outs[x-1], weight_dimension[x].rows, weight_dimension[x].cols);
            z_outs[x] = Z;
            printf("\n\n");
        }
    }

    //print z layers.
    for (int z_l = 0; z_l < no_layer; z_l++){
        for (int r = 0; r < 100; r++){
            for (int c = 0; c < weight_dimension[z_l].cols; c++){
                printf("%f,%s", z_outs[z_l][r][c]," ");
            } 
            printf("\n");
        } 
        printf("\n\n");
    } 

    //activate the z layers 

    printf("\n\n------------------------- acivation_output from each output -------------------------\n\n\n");
    
    for (int z_l = 0; z_l < no_layer-1; z_l++){
        double ** activated_l = activate_zlayer(z_outs[z_l],weight_dimension[z_l].cols,z_l);
        a_outs[z_l] = activated_l;
    } 

    for (int a_l = 0; a_l < no_layer-1; a_l++){
        for (int r = 0; r < 100; r++){
            for (int c = 0; c < weight_dimension[a_l].cols; c++){
                printf("%f,%s", a_outs[a_l][r][c]," ");
            } 
            printf("\n");
        } 
        printf("\n\n");
    }

    //CALCULATE THE COST
    double* cost = sparse_cat_loss_vector(a_outs[sizeof(a_outs)/sizeof(a_outs[0])-1], y_data, 100,2);


    //TRAIN THE NEURAL NETWORK.
    //double ***gradients = calculate_gradients(weights, bias, a_outs, z_outs, model_structure, weight_dimension);

    /*int epochs = 100;
    for (int i = 0; i < epochs; i++){
        double ***gradients = calculate_gradients(weights,bias,a_outs[sizeof(a_outs)/sizeof(a_outs[0])-1],model_structure,weight_dimensions);

    }*/




    





    



    



    


    return 0;
}
