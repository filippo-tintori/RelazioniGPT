module blinkIFswitch (
    input CLK100MHZ,
    input sw,
    output reg [15:0] LED
);

reg [32:0] counter;

initial 
    counter = 0, LED = 0;

always @(posedge CLK100MHZ) begin
    if (sw) begin
        LED[0] <= counter[23];
        counter <= counter + 1
    end else begin
        LED[0] <= counter[22]; # raddoppio se aperto
        counter <= counter + 1
        end
    end
endmodule