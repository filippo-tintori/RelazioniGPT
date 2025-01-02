module calcolatrice(
    input CLK100MHZ,
    input wire [15:0] sw,
    input btnL, btnU, btnR, btnD, btnC,
    output reg[15:0] LED
);

reg [15:0] n1,n2;
reg [15:0] Cb, Lb, Rb, Ub, Db;
reg press;

initial begin
    n1=0, n2=0, cB=0, Lb=0; Rb=0, Ub=0, Db=0, press=0, LED=0;
end

always @(posedge CLK100MHZ) begin
    Lb <= btnL;
    Rb <= btnR;
    Ub <= btnU;
    Db <= btnD;
    Cb <= btnC;

    if ( (btnC == 0) && (Cb == 1) && (press == 0) ) begin
        n1 <= sw;
        LED <= sw;
        press <= 1;
    end

    if ( (btnC == 0) && (Cb == 1 ) && (press == 1)) begin
        n2 <= sw;
        LED <= sw;
        press <= 0;
    end

    if ( (btnL == 0) && (Lb == 1) ) begin
        LED <= n1 + n2;
    end

    if ( (btnR == 0) && (Lb == 1) ) begin
        LED <= n1 - n2;
    end

    if ( (btnD == 0) && (Db == 1) ) begin
        LED <= n1 * n2;
    end

    if ( (btnU == 0) && (Ub == 1) ) begin
        if (n2 != 0) begin
            LED <= n1 / n2;
        end else begin
            LED <= 16'hFFFF;
            press <=0;
        end
        
    end
end

endmodule