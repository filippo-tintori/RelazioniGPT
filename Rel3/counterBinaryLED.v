module counterBinaryLED (
    input CLK100MHZ
    output [15:0] LED
);

reg [32:0] counter;
reg i;
always @(posedge CLK100MHZ) begin
    for (i = 0; i<16; i = i+1) begin
        LED[i] < counter [22-i];
    end
end
    
endmodule