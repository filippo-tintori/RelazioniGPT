module LEDmanualePACMAN (
    input CLK100MHZ,
    input btnL, btnR,
    output reg [15:0] LED
);
reg [3:0] indled;
reg obtnL, obtnR;

initial begin
    indled = 0;
    LED = 16'b0000_0000_0000_0001;
    obtnL = 0;
    obtnR = 0;
end

always @(posedge CLK100MHZ) begin
    obtnL <= btnL;
    obtnR <= btnR;

    if (btnL && !obtnL) begin
        indled <= (indled == 15) ? 0 : indled +1;
    end else if (btnR && !obtnR) begin
        indled <= (indled == 0) ? 15 : indled -1;
    end
    LED <= 16'b0 << indled;
    LED[indled] <= 1
end

endmodule