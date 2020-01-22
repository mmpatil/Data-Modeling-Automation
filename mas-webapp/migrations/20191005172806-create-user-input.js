'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('UserInput', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
            model: 'RunDetail',
            key: 'Id'
        },
        onUpdate: 'cascade',
        onDelete: 'cascade'
      },
      UserId: Sequelize.INTEGER,
      DWLimitLow: Sequelize.FLOAT,
      DWLimitHigh: Sequelize.FLOAT,
      BGLimit: Sequelize.FLOAT,
      WhiteSkedacityLimit: Sequelize.FLOAT,
      VIFLimit: Sequelize.FLOAT,
      ADFLimit: Sequelize.FLOAT,
      DynamicBacktestRange1: Sequelize.DATE,
      DynamicBacktestRange2: Sequelize.DATE,
      DynamicBacktestRange3: Sequelize.DATE,
      DynamicBacktestRange4: Sequelize.DATE,
      DynamicBacktestRange5: Sequelize.DATE,
      DynamicBacktestRange6: Sequelize.DATE,
      DynamicBacktestRange7: Sequelize.DATE,
      DynamicBacktestRange8: Sequelize.DATE,
      DynamicBacktestRange9: Sequelize.DATE,
      DynamicBacktestRange10: Sequelize.DATE,
      DynamicBacktest1Weight: Sequelize.FLOAT,
      DynamicBacktest2Weight: Sequelize.FLOAT,
      DynamicBacktest3Weight: Sequelize.FLOAT,
      DynamicBacktest4Weight: Sequelize.FLOAT,
      DynamicBacktest5Weight: Sequelize.FLOAT,
      DynamicBacktest6Weight: Sequelize.FLOAT,
      DynamicBacktest7Weight: Sequelize.FLOAT,
      DynamicBacktest8Weight: Sequelize.FLOAT,
      DynamicBacktest9Weight: Sequelize.FLOAT,
      DynamicBacktest10Weight: Sequelize.FLOAT,
      DynamicBacktestLongRange1: Sequelize.DATE,
      DynamicBacktestLongRange2: Sequelize.DATE,
      DynamicBacktestLongRange3: Sequelize.DATE,
      DynamicBacktestLongRange4: Sequelize.DATE,
      DynamicBacktestLongRange5: Sequelize.DATE,
      DynamicBacktestLongRange6: Sequelize.DATE,
      DynamicBacktestLongRange7: Sequelize.DATE,
      DynamicBacktestLongRange8: Sequelize.DATE,
      DynamicBacktestLongRange9: Sequelize.DATE,
      DynamicBacktestLongRange10: Sequelize.DATE,
      StatsTest: Sequelize.STRING,
      DependentTransformationType: Sequelize.STRING,
      DependentVariable: Sequelize.STRING
    });
  },

  down: (queryInterface, Sequelize) => {
      return queryInterface.dropTable('UserInput');
  }
};
