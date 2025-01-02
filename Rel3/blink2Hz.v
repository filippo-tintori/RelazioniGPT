module blink2Hz (
    input CLK100MHZ,
    output reg [15:0] LED;
);
    reg [32:0] counter;
    initial
        counter = 0;
always @(posedge CLK100MHZ) begin
    LED[0] <= counter[23];
    counter <= counter + 1;
end

endmodule