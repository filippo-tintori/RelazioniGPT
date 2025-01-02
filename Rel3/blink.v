module blink (
    input CLK100MHZ,
    output reg [15:0] LED
);

reg counter;

initial 
    counter = 0;

always @(posedge CLK100MHZ) begin
    LED[0] <= counter;
    counter <= counter + 1;
end

endmodule